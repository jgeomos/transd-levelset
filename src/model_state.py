"""
Model state management for geological inversion.

This module provides ModelStateManager, a class for tracking geometrical
and property models across units during inversion.

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

import numpy as np
import src.utils.transd_utils as nu


class ModelStateManager:
    """
    Manages geometric model state and level set representations during inversion.
    
    Maintains current, previous, and temporary model states along with their
    corresponding signed distance (level set) functions. Supports model updates,
    reversions, and domain masking for constrained perturbations.
    
    In level set methods, the geometric model (m_curr) is derived from signed
    distance functions (phi_curr). Each geological unit has a signed distance
    field where positive values indicate inside the unit and negative values
    indicate outside.
    
    Attributes
    ----------
    m_curr : ndarray
        Current density/property model (3D)
    m_prev : ndarray
        Previous accepted model for revert operations
    m_tmp, m_tmp_1, m_tmp_2 : ndarray
        Temporary models for trial calculations
    phi_curr : ndarray
        Current signed distance functions (n_units × nx × ny × nz)
    phi_prev : ndarray
        Previous signed distance functions
    phi_orig : ndarray
        Original unperturbed signed distances
    mod_aux : ndarray
        Auxiliary model for forced perturbations
    phi_aux : ndarray
        Signed distances for auxiliary model
    domain_mask : ndarray
        Mask controlling where perturbations can occur (1=allowed, 0=masked)
    dim : tuple
        Model dimensions (nx, ny, nz)
    
    Notes
    -----
    The level set framework works as follows:
    1. Each unit has a signed distance field in phi_curr
    2. At each cell, the unit with maximum signed distance is assigned
    3. m_curr = petrovals.all_values[argmax(phi_curr, axis=0)]
    4. Perturbations modify phi_curr, then m_curr is regenerated
    
    State transitions:
    - propose → accept: m_prev ← m_curr, phi_prev ← phi_curr
    - propose → reject: m_curr ← m_prev, phi_curr ← phi_prev (revert)
    
    Examples
    --------
    >>> # Initialize
    >>> mvars = ModelStateManager(dim=(50, 60, 40))
    >>> mvars.m_curr = initial_density_model
    >>> 
    >>> # Initialize signed distances
    >>> mvars.init_signed_distances(petrovals, cell_size=[100, 100, 50])
    >>> 
    >>> # Initialize temporary models
    >>> mvars.init_tmp()
    >>> 
    >>> # Propose perturbation
    >>> mvars.phi_curr[2] += perturbation  # Modify unit 2
    >>> mvars.m_curr = nu.get_density_model(petrovals.all_values, mvars.phi_curr)
    >>> 
    >>> # Accept or reject
    >>> if accepted:
    ...     mvars.update_prev(accepted=True)
    ... else:
    ...     mvars.revert_to_prev()
    
    See Also
    --------
    PetroStateManager : Manages petrophysical property values
    calc_signed_distances : Computes signed distance functions
    get_density_model : Derives model from signed distances
    """
    
    # Define allowed attributes
    __slots__ = [
        'm_curr', 'm_prev', 'm_tmp', 'm_tmp_1', 'm_tmp_2', 'm_nullspace_last', 
        'domain_mask', 'loaded_mask',
        'm_start', 'mod_aux', 'mod_aux_prev', 'delta_m_orig', 'delta_m',
        'phi_orig', 'phi_curr', 'phi_prev', 'phi_start', 'phi_prior',
        'phi_aux', 'phi_aux_prev', 'delta_phi', 'index_unit_pert', 'dim'
    ]

    def __init__(self, dim: np.ndarray = None):
        """
        Initialize model state manager.
        
        Parameters
        ----------
        dim : tuple of int, optional
            Model dimensions (nx, ny, nz). If provided, sets self.dim.
        """
        # Current model of null space navigation, all model-cells.
        self.m_curr = None
        # Previous model.
        self.m_prev = None
        # Temporary model for calculation of trial models.
        self.m_tmp = None
        self.m_tmp_1 = None
        self.m_tmp_2 = None
        # Full model with perturbation added of complete space navigation.
        self.m_nullspace_last = 0.

        # Mask of the indices of subset within full model that are considered for model perturbations.
        self.domain_mask = None
        # Mask to be loaded from external file -- 0s or 1s to control where changes can occur.
        self.loaded_mask = None

        # Initial model before perturbation.
        self.m_start = None

        # Auxiliary model used for forced pert.
        self.mod_aux = None
        self.mod_aux_prev = None

        # First perturbation of model.
        self.delta_m_orig = None

        # Perturbation of current model.
        self.delta_m = None

        # Full unperturbed initial signed distances for beginning of model perturbations.
        self.phi_orig = None

        # Current signed distances during model perturbations.
        self.phi_curr = None

        # Previous signed distances.
        self.phi_prev = None

        # Starting signed distances.
        self.phi_start = None  # TODO: remove - not used?

        # Starting signed distances.
        self.phi_prior = None

        # Signed distances for auxiliary model.
        self.phi_aux = None
        self.phi_aux_prev = None

        # Perturbation of current signed distances.
        self.delta_phi = None

        # Index of perturbed unit.
        self.index_unit_pert = None

        # Dimension of 3D model.
        self.dim = dim

        if dim is not None: 
            self.dim = dim

    def init_tmp(self):
        """
        Initialize temporary model arrays from current model.
        
        Creates copies of m_curr in m_tmp, m_tmp_1, and m_tmp_2 for use in
        trial calculations without modifying the current model.
        
        Raises
        ------
        ValueError
            If m_curr is None (must be set before calling init_tmp)
        """
        if self.m_curr is not None: 
            self.m_tmp = np.copy(self.m_curr)
            self.m_tmp_1 = np.copy(self.m_curr)
            self.m_tmp_2 = np.copy(self.m_curr)
        else: 
            raise ValueError("init_tmp is to init the tmp models using m_curr. Does not work when m_curr is None!!")
    
    def init_signed_distances(self, petrovals, cell_size=[1.,1.,1.]):
        """
        Calculate initial signed distance functions for all units.
        
        Computes level set representations for the current model and auxiliary
        model (if present). Sets up phi_curr, phi_start, phi_prior, and phi_aux.
        
        Parameters
        ----------
        petrovals : PetroStateManager
            Petrophysical state with unit values
        cell_size : list of float, default=[1., 1., 1.]
            Cell dimensions [dx, dy, dz] in meters
            
        Notes
        -----
        For main model, computes signed distances for all units in petrovals.all_values.
        For auxiliary model, computes distances only for perturbation units.
        """
        
        # Get the signed distances of the perturbation. 
        # mvars.phi_nullspace_orig = nu.calc_signed_distances(mvars.delta_m_orig, petrovals.pert, cell_size=cell_size)
        if self.mod_aux is not None: 
            self.phi_aux = nu.calc_signed_distances(self.mod_aux, 
                                                    petrovals.pert[petrovals.pert != 0.], 
                                                    cell_size=cell_size).astype(np.float32)
            self.phi_aux_prev = self.phi_aux.copy()
        
        # Get the signed distances of the model with initial perturbation. 
        self.phi_curr = nu.calc_signed_distances(self.m_curr, petrovals.all_values, cell_size=cell_size).astype(np.float32)
        self.phi_start = self.phi_curr.copy()
        self.phi_prior = self.phi_curr.copy()

    def init_prev_model(self):        
        """
        Initialize previous state by copying current state.
        """
        self.m_prev = self.m_curr.copy() 
        self.phi_prev = self.phi_curr.copy()
        # aux models: used only for forced perturbation and nullspace navigation.
        if self.mod_aux_prev is not None: 
            self.mod_aux_prev = self.mod_aux.copy()
            self.phi_aux_prev = self.phi_aux.copy()
        else: 
            self.mod_aux_prev = None 
            self.phi_aux_prev = None

    def update_prev(self, accepted):
        """ 
        Update previous state if proposal was accepted. 
        """
        if accepted: 
            np.copyto(self.m_prev, self.m_curr)
            # Case where dimensions change due to birth or death. 
            if np.shape(self.phi_prev) == np.shape(self.phi_curr):
                np.copyto(self.phi_prev, self.phi_curr)
            else: 
                self.phi_prev = self.phi_curr.copy()
            if self.mod_aux_prev is not None: 
                np.copyto(self.mod_aux_prev, self.mod_aux)
                np.copyto(self.phi_aux_prev, self.phi_aux) 

    def revert_to_prev(self): 
        """ 
        Revert current state to previous state (reject proposal). 
        """
        np.copyto(self.m_curr, self.m_prev) 
        if self.mod_aux is not None:
            np.copyto(self.mod_aux, self.mod_aux_prev)  
            np.copyto(self.phi_aux, self.phi_aux_prev)
        if np.shape(self.phi_curr) == np.shape(self.phi_prev):
            np.copyto(self.phi_curr, self.phi_prev)
        else: 
            self.phi_curr = self.phi_prev.copy()

    def set_masked_domain(self, use_mask_domain, distance_max, ind_unit_mask, dens_model, mask_first_layers):
        """
        Create mask controlling where model perturbations can occur based on distance to 
        a reference unit's interface.

        Can also mask the top two layers of the model. 

        Perturbations will only be applied
        where domain_mask == 1.
        
        Parameters
        ----------
        use_mask_domain : bool
            If True, calculate distance-based mask. If False, allow changes everywhere.
        distance_max : float
            Maximum distance (in cells) from reference unit interface to allow changes
        ind_unit_mask : int
            Index of reference unit for distance calculation
        dens_model : ndarray
            Current density model (3D)
        mask_first_layers : bool
            If True, mask the two shallowest layers (assumed well-constrained)
        """

        def calc_mask_dist(phi_, distance_max_, unit_index_):
            """
            Set the mask on the distance to the interface of selected rock unit.
            
            Parameters
            ----------
            phi_ : ndarray
                Signed distance fields (n_units × nz × nx × ny)
            distance_max_ : float
                Distance threshold for calculation of mask
            unit_index_ : int
                Index of rock unit to calculate distance from
                
            Returns: ndarray
            """

            phi_mask = phi_[unit_index_][:, :, :].copy()
            phi_mask[phi_[unit_index_][:, :, :] > - distance_max_] = 1.
            phi_mask[phi_[unit_index_][:, :, :] < - distance_max_] = 0.

            return phi_mask

        def calc_mask_layer(masked_layer, horizontal_layer_index):
            """
            Set the layer of model cells at given depth index to 0 (masked).
            """
            masked_layer[horizontal_layer_index, :, :] = 0.

        assert dens_model.ndim == 3, "Density contrast model array should not be flat. It should be a 3D array."

        # Calculates the mask based on distance to outline of selected units.
        if use_mask_domain:
            # Calculate the signed distances using the density model: used to define mask controlling areas that can change.
            phi = nu.calc_signed_distances(dens_model, np.unique(dens_model), cell_size=None, narrow=False)

            # Mask on cells farther than a certain distance to the units with index ind_unit_mask
            # In the paper example: masks values further than a certain distance away from the mantle.
            self.domain_mask = calc_mask_dist(phi, distance_max, unit_index_=ind_unit_mask)

            # Apply mask by default to the 2 shallowest units: assuming they are well constrained.
            calc_mask_layer(self.domain_mask, horizontal_layer_index=0)
            calc_mask_layer(self.domain_mask, horizontal_layer_index=1)

        else:
            self.domain_mask = np.ones_like(dens_model)
            
        # Hardcoded constraints. TODO: remove hardcoded constraints.
        if mask_first_layers:
            # Used for the example, not the Pyrenees case.
            self.domain_mask[0, :, :] = 0.  # First layer.
            self.domain_mask[1, :, :] = 0.  # Second layer.
            # mask_modelling_domain[2, :, :] = 0.  # Third layer.

        self.domain_mask = self.domain_mask.flatten()

    def assert_shape_dtype(self, DEBUG):
        """
        Assert consistency of array shapes and dtypes for debugging.
        
        Parameters
        ----------
        DEBUG : bool.   If True, perform assertions. If False, no-op.
           
        Raises: AssertionError, If any shape or dtype mismatches are found
        """
        if DEBUG:
            assert self.m_prev.shape == self.m_curr.shape, f"Shape mismatch: {self.m_prev.shape} vs {self.m_curr.shape}, Fail assert 1"
            assert self.m_prev.dtype == self.m_curr.dtype, f"Dtype mismatch: {self.m_prev.dtype} vs {self.m_curr.dtype}, Fail assert 2"
            # assert self.phi_prev.shape == self.phi_curr.shape, f"Shape mismatch: {self.phi_prev.shape} vs {self.phi_curr.shape}, Fail assert 3" 
            assert self.phi_prev.dtype == self.phi_curr.dtype, f"Dtype mismatch: {self.phi_prev.dtype} vs {self.phi_curr.dtype}, Fail assert 4"  
            if self.mod_aux is not None:
                assert self.mod_aux.shape == self.mod_aux_prev.shape, f"Shape mismatch: {self.mod_aux.shape} vs {self.mod_aux_prev.shape}, Fail assert 5"
                assert self.mod_aux.dtype == self.mod_aux_prev.dtype, f"Dtype mismatch: {self.mod_aux.dtype} vs {self.mod_aux_prev.dtype}, Fail assert 6"
                assert self.phi_aux.shape == self.phi_aux_prev.shape, f"Shape mismatch: {self.phi_aux.shape} vs {self.phi_aux_prev.shape}, Fail assert 7"
                assert self.phi_aux_prev.dtype == self.phi_aux.dtype, f"Dtype mismatch: {self.phi_aux_prev.dtype} vs {self.phi_aux.dtype}, Fail assert 8"  
