"""
Birth–death configuration management for geological inversion.

This module provides configuration classes used to control element creation 
("birth") and removal ("death") processes within geological inversion or 
model evolution algorithms. These classes encapsulate all parameters required 
to initialize, validate, and manage the nucleation and annihilation of 
geological units or model components.

Classes
-------
BirthConfig
    Configuration parameters governing the nucleation (birth) of new 
    geological units or density domains. Handles setup, validation, and 
    gradient-based threshold control for candidate regions.
DeathConfig
    Configuration parameters controlling the removal (death) of existing 
    geological units or model components, ensuring consistency between 
    model state and tracking arrays.

Authors:
    Jérémie Giraud
    Vitaliy Ogarko
    
License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

import numpy as np
import src.utils.quality_control as qc
import src.utils.transd_utils as tu


class BirthConfig:
    """
    Configuration parameters for geological unit nucleation. 
    Default parameters are suitable to most cases. 
    Encapsulates parameters needed for the birth/nucleation process,
    """
    # TODO move the params to shpars and keep only what varies here? 

    # Define allowed attributes.
    __slots__ = ['n_blobs_max', 'n_blobs_curr', 'n_line_search', 'min_cells_num', 'flag_conform_geol', 
                 'mask_max_grad', 'max_grad_ind', 'bounds', '_rng_seed', 'nucl_geol', 'n_births_max',
                 'ind_nucl', 'type_birth', 'exclude_indices', 'domains_births', 'grad_thresh']

    # TODO move default values to expert parameters in parfile?

    def __init__(self, 
                n_blobs_max: int = 4,
                n_blobs_curr: int = 0,
                n_line_search: int = 10, 
                min_cells_num: int = 100, 
                flag_conform_geol: int = 2, 
                model_dim: tuple = None, 
                bounds: tuple = (-100, 100), 
                rng_seed: object = None, 
                nucl_geol: int = 2, 
                n_births_max: int = 5,
                type_birth: int = 0, 
                exclude_indices: np.array = None, 
                domains_births: np.array = None):

        """
        Initialize nucleation parameters.
        
        Args:
            n_blobs_max: Maximum number of groups/blobs to consider for births
            n_line_search: Number of gradient threshold values to test
            min_cells_num: Minimum number of contiguous cells for nucleation
            bounds: Petrophy values bounds for optimization (min, max)
            rng_seed: Random seed for reproducible threshold generation
            nucl_geol: Geological nucleation mode:
                - 0: No domains, whole volume can be used
                - 1: Domains follow units in groups
                - 2: Domains follow all rock units
            type_birth: Birth type:
                - 0: Deterministic linearly spaced thresholds
                - other: Random uniform thresholds
            exclude_indices: Indices to exclude from nucleation (tuple/array/None)
            
        """
        self.n_blobs_max = n_blobs_max  
        self.n_blobs_curr = n_blobs_curr  
        self.n_line_search = n_line_search
        self.min_cells_num = min_cells_num
        self.flag_conform_geol = flag_conform_geol
        self.mask_max_grad = np.zeros(model_dim, dtype=np.int32)
        self.max_grad_ind = None
        self.bounds = bounds
        self._rng_seed = rng_seed
        self.nucl_geol = nucl_geol
        self.n_births_max = n_births_max
        self.ind_nucl = None
        self.type_birth = type_birth
        self.exclude_indices = exclude_indices
        self.domains_births = domains_births
        self.grad_thresh = None

        # QC that values are reasonable. 
        self.validate()
        

    def validate(self):
        """Validate parameters with detailed error messages."""
        errors = []
        
        if self.n_blobs_max <= 0:
            errors.append(
                f"n_blobs_max must be positive, got {self.n_blobs_max}\n"
                f"  This controls how many gradient regions to test for births.\n"
                f"  Recommended: min 3-5 for typical cases"
                        )
        if self.n_line_search <= 0:
            errors.append(f"n_line_search must be positive, got {self.n_line_search}")
        
        if self.min_cells_num <= 0:
            errors.append(f"min_cells_num must be positive, got {self.min_cells_num}")
        
        if len(self.bounds) != 2 or self.bounds[0] >= self.bounds[1]:
            errors.append(f"bounds must be (min, max) with min < max, got {self.bounds}")
        
        if self.nucl_geol not in [0, 1, 2]:
            errors.append(f"nucl_geol must be 0, 1, or 2, got {self.nucl_geol}")
        
        if errors:
            raise qc.BirthParameterValidationError(
                f"Invalid BirthVariables configuration:\n" + 
                "\n".join(f"  - {error}" for error in errors)
            )
    
    def set_gradient_thresholds(self):
        """Initialize threshold for gradient of cost function to test for geological unit nucleation."""
        
        # Generate gradient thresholds
        if self.type_birth == 0:
            self.grad_thresh = np.linspace(1 / self.n_line_search, 1, self.n_line_search)
        else:
            self.grad_thresh = self._rng_seed.uniform(1 / self.n_line_search, 1, self.n_line_search) 

    def calc_births_domains(self, model_curr):
        """
        flag_conform_geol: decides whether the nucleation will conform to the existing geological model.
        This function is now general but was originally designed for application to the case using data from the Pyrenees.
        Arguments 'mantle', 'crust', 'shallow' and 'isolated' refer to domains corresponding to rock group of rock units
        that cover several smaller units from the original rock model.

        # 1. flag_conform_geol == 0. No domains: geometry of units/group of units not accounted for, the whole volume can be used (not supported).
        # 2. flag_conform_geol == 1. Domains follow units put together in groups (not supported).
        # 3. flag_conform_geol == 2. Domains follow all rock units.

        :param model:
            :param flag_conform_geol:
        :return:
        """

        # Get the different petrophysical values populating the model.
        petro_values = tu.get_petro_values(model_curr)  # This line creates a dependency. Should this be removed? 

        # Follow all rock units in model.
        if self.flag_conform_geol == 2:
            self.domains_births = np.ones_like(model_curr)
            for i in range(len(petro_values)):
                self.domains_births[model_curr == petro_values[i]] = i
        else: 
            raise qc.ParameterValidationError("`flag_conform_geol` must be 2. Other values are currently not supported.")
        # TODO allow the case of some specific values based on groups of units (self.flag_conform_geol=1).
