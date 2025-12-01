"""
Inversion metrics tracking and monitoring.

This module provides the InversionMetrics class for tracking and recording
metrics during geological inversion, including acceptance rates, prior ratios,
data misfit evolution, and convergence diagnostics.

Classes
-------
InversionMetrics
    Main class for tracking inversion progress and quality metrics

Authors:
    Jérémie Giraud

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"


# @njit(fastmath=True) 
# def calc_prior_cost(signdist_, reference_signdist_, std_geom_glob_, weights_phi_):
#     """Optimized single cost calculation"""
#     diff = signdist_ - reference_signdist_
#     cost = (diff ** 2 / (std_geom_glob_ ** 2)) * (weights_phi_ ** 2)
#     return np.sum(cost) / cost.size

# @njit(fastmath=True) # Using numba makes this slower -- fix this!!! Is it because this function is called inside a class?
# def calc_log_priorgeom_ratio(current_signdist, previous_signdist, reference_signdist, std_geom_glob, inv_var):
#     """Use float32 to halve memory bandwidth"""
    
#     # Convert to float32 (half the memory bandwidth)
#     curr = current_signdist.astype(np.float32)
#     prev = previous_signdist.astype(np.float32) 
#     ref = reference_signdist.astype(np.float32)
    
#     cost_curr = np.sum(np.square(curr - ref) * inv_var) / curr.size  # TODO incorporate the division in inv_var?
#     cost_prev = np.sum(np.square(prev - ref) * inv_var) / prev.size
    
#     return float(-cost_curr + cost_prev), float(np.sqrt(cost_curr))

import numpy as np
import src.utils.quality_control as qc
from numba import jit
import math


@jit(nopython=True, cache=True)
def calc_rms(vector_):
    # This function is the same as calc_rms in foward calculation utils module.
    """RMS calculation from vector. """
    sum_sq = 0.0
    n = len(vector_)
    for i in range(n):
        sum_sq += vector_[i] * vector_[i]
    return math.sqrt(sum_sq / n)


class InversionMetrics:
    """
    A class containing metrics for the monitoring of sampling and navigation.
    """

    # Class constant.
    MAX_LOG_RATIO = 30.0  # For calc of likelihood ratio, numerical purpose. Beyond this, exp() underflows to 0

    # Define allowed attributes.
    __slots__ = ['reference_misfit', 'max_misfit_force', 'data_misfit', 
                'model_misfit', 'petro_misfit', 'accept_ratio', 
                'log_likelihood_ratio', 'log_likelihood', 'log_priorgeom_ratio', 'log_priorgeom', 
                'log_priorpetro_ratio', 'log_priorpetro', 'log_posterior',
                'it_accepted_model', 'it_accepted_type', 'last_misfit_accepted', 'contact_areas',
                'n_units_total']

    def __init__(self, reference_misfit, starting_misfit, n_proposals=1, max_misfit_force=np.inf):
        # Validation
        if n_proposals <= 0:
            raise qc.ParameterValidationError(f"`n_proposals` must be positive. Got: {n_proposals}")
        if reference_misfit < 0:
            raise qc.ParameterValidationError(f"`reference_misfit` cannot be negative. Got: {reference_misfit}")
        if max_misfit_force < 0:
            raise qc.ParameterValidationError(f"`max_misfit_force` cannot be negative. Got: {max_misfit_force}")

        # Related to misfits. 
        self.reference_misfit: float = reference_misfit
        self.max_misfit_force: float = max_misfit_force

        self.data_misfit: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.model_misfit: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.petro_misfit: np.ndarray = np.zeros(n_proposals, dtype=np.float32)

        # Related to posterior and acceptance. 
        self.accept_ratio: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_likelihood_ratio: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_likelihood: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_priorgeom_ratio: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_priorgeom: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_priorpetro_ratio: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_priorpetro: np.ndarray = np.zeros(n_proposals, dtype=np.float32)
        self.log_posterior: np.ndarray = np.zeros(n_proposals, dtype=np.float32)

        self.it_accepted_model: list = []
        self.it_accepted_type: list = []

        self.n_units_total = np.zeros(n_proposals, dtype=int)

        # Set initial values
        self.last_misfit_accepted: float = starting_misfit
        self.data_misfit[0] = starting_misfit

        # Track contact evolution
        self.contact_areas = [] 

    # def track_contacts(self, model, cell_sizes, iteration):  # TODO is this proper writing, should it be there? 
    #     """Track contact surface areas over iterations."""
    #     contacts = nu.calculate_unit_contacts_fast(model, cell_sizes)
    #     self.contact_areas.append({
    #         'iteration': iteration,
    #         'contacts': contacts,
    #         'total_area': sum(contacts.values())
    #     })
    #     return contacts

    def calc_data_rms_misfit(self, geophy_data, ind):
        """ Calculate the root-mean-square data misfit using the residuals vector. """
        self.data_misfit[ind] = calc_rms(geophy_data.data_field - geophy_data.data_calc)
    
    @staticmethod
    def calc_geometric_prior_sq_difference(phi_curr, phi_prior, petrovals, inv_var, mode='l2_norm'):
        """
        Calculate difference between current and prior signed distances for original units.
        
        Computes the deviation of current geometric configuration from the prior
        (initial) model for original units only. 

        For birthed units, calculation is done separately in another function.
        
        Parameters
        ----------
        phi_curr: current signed distances
        phi_prior: prior signed distances
        petrovals : PetroStateManager
            Petrophysical state for identifying original units
        inv_var: np.array
            weights applied to the difference between phi_curr and phi_prior
        mode : str, default='l2_norm'
            Method for calculating difference:
            - 'l2_norm': Euclidean norm (sum of squares)
            - 'l1_norm': L1 norm TODO.
            - 'l0_norm': TODO cf VO discussion with Vitaliy. 
            
        Returns
        -------
        float or ndarray
            Geometric prior difference. 
            
        Notes
        -----
        Only considers original units (origin == 'orig'). Birthed units have no
        prior and are excluded from calculation.
        
        The signed distance difference measures how much the geometry has changed
        from the prior/initial model configuration.
        
        Examples
        --------
        >>> # Calculate L2 norm of geometric deviation
        >>> prior_misfit = calc_geometric_prior_difference(mvars, petrovals)
        """

        # Identify indices of original units
        original_indices = []
        current_indices = []
        for i, (val, item_id, origin, _, _, orig_idx) in enumerate(petrovals._tracked_state["items"]):
            if origin == 'orig':
                original_indices.append(orig_idx)
                current_indices.append(i)

        if len(original_indices) == 0:
            raise qc.CodeBug(f"NO ORIGINAL UNITS FOUND!!!!!")
            # return 0.0
        
        # Extract signed distances for original units and calculate differences.
        diff = phi_curr[current_indices] - phi_prior[original_indices]


        
        # Calc value of prior model misfit term.
        # sum = 0.
        if mode == 'l2_norm':
        # Flatten spatial dimensions of diff. 
            n_units = diff.shape[0]
            diff = diff.reshape(n_units, -1) 
            # In-place calc. 
            diff *= inv_var
            diff = np.square(diff, out=diff)
            return np.sum(diff)
        else: 
            raise qc.ParameterValidationError(f"Unknown mode: {mode}")
        
    @staticmethod
    def calc_birthed_units_penalty(phi_curr, petrovals, inv_var, birth_penalty_weight=1., mode='l2_norm'):
        """
        Calculate penalty term for birthed units based on signed distance magnitudes.
        A prior value of 0 is assumed. 
        
        Birthed units have no prior, so their existence is penalised based on the
        magnitude of their signed distances. In a nutshell, this discourages excessive 
        nucleation while allowing births that improve data fit.
        
        Parameters
        ----------
        phi_curr : ndarray
            Current signed distances (n_units × nx × ny × nz)
        petrovals : PetroStateManager
            Petrophysical state for identifying birthed units
        inv_var: np.array
            weights applied to the signed distance of birthed unit.
        mode : str, default='l2_norm'
            Method for calculating penalty:
            - 'l2_norm': Euclidean norm (sum of squares)
            # - 'l1_norm': Manhattan norm (sum of absolute values) TODO
            # - 'l0_norm': TODO
            # - 'volume': Count of cells occupied (volume penalty)
            
        Returns
        -------
        float
            Penalty value. 
            
        Notes
        -----
        Only considers birthed units (origin == 'birth').
        
        The penalty represents how much "presence" birthed units have in the model.
        Higher positive signed distances = more cells occupied = higher penalty.
        
        Examples
        --------
        >>> # Calculate L2 penalty for all birthed units
        >>> birthed_unit_cost = calc_birthed_units_penalty(
        ...     mvars.phi_curr, petrovals, birth_penalty_weight, mode='L2_norm')
        """       
        # Identify indices of birthed units in phi_curr
        birthed_indices = []
        for i, (val, item_id, origin, _, _, orig_idx) in enumerate(petrovals._tracked_state["items"]):
            if origin == 'birth':
                birthed_indices.append(i)
        
        # No birthed units - no penalty
        if len(birthed_indices) == 0:
            return 0.0
        
        # Extract signed distances for birthed units
        phi_birthed = birth_penalty_weight * phi_curr[birthed_indices]
        
        # Calculate term. 
        if mode == 'l2_norm':
            # Sum the individual contribution of each unit. 
            n_birthed_units = phi_birthed.shape[0]
            phi_birthed = phi_birthed.reshape(n_birthed_units, -1) 
            # In-place calc. 
            phi_birthed = np.square(phi_birthed, out=phi_birthed)
            return np.sum(phi_birthed)
        else: 
            raise qc.ParameterValidationError(f"Unknown mode: '{mode}'. Use 'l2_norm for now!")
        # elif mode == 'l0_norm':
        # elif mode == 'l1_norm':
        #     # Sum of absolute signed distances
        #     return float(np.sum(np.abs(phi_birthed)))
        # elif mode == 'volume':
        #     # Count cells where birthed units are present (phi > 0)
        #     n_cells = np.sum(phi_birthed > 0)
        #     return float(n_cells)
    
    def calc_log_likelihood_ratio(self, std_data_misfit, ind, log_run=None):
        """ 
        Calculates log(likelihood ratio).
        Handles the case of an objective, or reference misfit value. 
        """
    
        current_misfit = self.data_misfit[ind]
        previous_misfit = self.last_misfit_accepted
        reference_misfit = self.reference_misfit

        # Log-likelihood ratio. >0 is favourable, and <0 is unfavorable. 
        sum_sq_prev = (previous_misfit - reference_misfit) ** 2
        sum_sq_curr = (current_misfit - reference_misfit) ** 2
        self.log_likelihood[ind] = (-sum_sq_curr) / std_data_misfit
        self.log_likelihood_ratio[ind] = (-sum_sq_curr +sum_sq_prev) / std_data_misfit

        # Logging.
        if log_run is not None:
            log_run.info(f'iteration {ind}: reference_misfit {reference_misfit:.6f}')
            log_run.info(f'iteration {ind}: current_misfit {current_misfit:.6f}')
            log_run.info(f'iteration {ind}: previous_misfit {previous_misfit:.6f}')
            log_run.info(f'iteration {ind}: log_likelihood_ratio {self.log_likelihood_ratio[ind]:.6f}')

        return None
    
    def calc_log_priorgeom_ratio(self, petrovals, current_signdist, reference_signdist, std_geom_glob, 
                                 inv_var, log_run, ind, pert_type):

        # inv_var: inverse variance -- calculated using standard dev and spatial weights.
        # TODO calculate based on previous -- if only one unit is different from previous, just update cost based on difference.

        birth_penalty_weight = 1.
        
        # Handle the case of a petrophy perturbation.
        if pert_type == 2: 
            if ind == 0: 
                return None
            else: 
                self.model_misfit[ind] = self.model_misfit[ind-1]
                self.log_priorgeom_ratio[ind] = 0.
            return None

        total_size = current_signdist.size

        # # Handle the case of the first iteration.
        # cost_curr_total = self.model_misfit[ind]
        # if ind > 0:
        #     cost_prev_total = self.model_misfit[ind-1]

        # Calculate contribution of the prior model misfit from prior units.
        self.model_misfit[ind] = self.calc_geometric_prior_sq_difference(current_signdist, 
                                                                         reference_signdist, 
                                                                         petrovals, 
                                                                         inv_var, 
                                                                         mode='l2_norm')
        
        # Calculate and add the contribution of the birthed units. 
        self.model_misfit[ind] += self.calc_birthed_units_penalty(current_signdist, 
                                                                 petrovals, 
                                                                 inv_var,
                                                                 birth_penalty_weight=birth_penalty_weight,  
                                                                 mode='l2_norm')
        
        # Normalise by the number of units. 
        self.model_misfit[ind] /= total_size

        # Apply global weight on prior geometrical model.
        self.model_misfit[ind] /= std_geom_glob

        # Get the log-prior. TODO: this is equal to model_misfit. Some with log-likelihood --> replace and refactor?
        self.log_priorgeom[ind] = self.model_misfit[ind]

        # TODO use inv_var_sub for local weighting!

        if self.model_misfit[ind] == 0.:
            if log_run is not None:
                log_run.info("~~~~~~~~~~~~~~ No signed distances misfit!!! ~~~~~~~~~~~~~~")
                        
        # cost_curr = cost_curr_total / total_size
        # cost_prev = cost_prev_total / total_size
        
        if ind > 0:
            self.log_priorgeom_ratio[ind] = - self.model_misfit[ind] + self.model_misfit[ind-1]
            # self.model_misfit[ind] = np.sqrt(cost_curr)
            if log_run is not None:
                log_run.info(f'iteration {ind}: Previous model cost: {self.model_misfit[ind-1]}')
        else: 
            self.log_priorgeom_ratio[ind] = -self.model_misfit[ind]
            if log_run is not None:
                log_run.info(f'iteration {ind}: Previous model cost: {0.}')
        if log_run is not None:
            log_run.info(f'iteration {ind}: Current model cost {self.model_misfit[ind]}')
            log_run.info(f'iteration {ind}: log_priorgeom_ratio {self.log_priorgeom_ratio[ind]}')

        # self.log_priorgeom_ratio[ind] = 0.

        return None
    
    @staticmethod
    # @njit(fastmath=True)  # This makes things slower due to overhead. TODO make it an external arithmetic operation in another module? 
    def calc_petro_cost(petro_orig, petro_curr, cov_petro):
        return np.sum(((petro_curr - petro_orig) / cov_petro) ** 2)

    def calc_log_priorpetro_ratio(self, petrovals, ind, log_run=None):
        """Calculate log ratio of petrophysical prior."""

        std_petro = petrovals.std_petro
        distro_type = petrovals.distro_type

        if distro_type != 'gaussian':
            # Non-Gaussian not implemented - use uniform (ratio = 1)
            self.log_priorpetro_ratio[ind] = 0.0  # log(1) = 0
            self.petro_misfit[ind] = 0.0
            qc.issue_warning(message=f'it {ind}: Non-Gaussian distro for petro prior!! - log prior ratio set to 1.0!',
                            callback_printer=log_run.warning if log_run else None)
            return None

        sq_diff_curr, sq_diff_prev = petrovals.diff_from_orig()  
        
        # 'Gaussian' prior: log(P) = -diff² / (std_petro)
        log_petro_curr = -sq_diff_curr / std_petro
        log_petro_prev = -sq_diff_prev / std_petro
        
        self.log_priorpetro_ratio[ind] = log_petro_curr - log_petro_prev
        self.petro_misfit[ind] = -log_petro_curr
        self.log_priorpetro_ratio[ind] = -log_petro_curr  # TODO refactor to separate misfits and log-prior and log-likelihood!

        if log_run is not None: 
            log_run.info(f'iteration {ind}: log_petro_prev {log_petro_prev}')
            log_run.info(f'iteration {ind}: log_petro_curr {log_petro_curr}')
            log_run.info(f'iteration {ind}: log_priorpetro_ratio {self.log_priorpetro_ratio[ind]}')
        return None
    
    def calc_log_posterior(self, ind, log_run=None):
        """
        Calculate the log posterior as the sum of the log of other terms of the posterior. 
        Prints and logs is log_run not None. 
        """
        self.log_posterior[ind] =   self.log_likelihood[ind] + \
                                    self.log_priorgeom[ind] + \
                                    self.log_priorpetro[ind]
        if log_run is not None: 
            log_run.info(f'iteration {ind}: log_posterior {self.log_posterior[ind]}')
        return None

    def accept_proposal(self, rng_, force_accept_ratio, ind, pert_type, override_force=False):
        """
        Calculate acceptance ratio and then decide whether to accept or reject a proposal,
        with the possibility to force accept using a rate defined a priori by User. 
        
        override_force : bool, default=False
            If True, force acceptance regardless of probability (for kill_too_weak)
        
        Returns
        -------
        accept : bool
            Whether proposal is accepted
        force_accept : bool
            Whether acceptance was forced (vs natural Metropolis-Hastings)
        """

        # Calculate acceptance ratio using MH criterion.
        self.calc_accept_ratio(ind)

        # Special case: external override (e.g., kill_too_weak)
        if override_force:
            accept = True
            force_accept = True
            return accept, force_accept

        # Determine if model accepted using the usual MH criterion. 
        if rng_.uniform(0, 1) <= self.accept_ratio[ind]:
            accept = True
            force_accept = False
            return accept, force_accept

        # Determine if model accepted anyway using the forced acceptance ratio. 
        # Get the forced acceptance rate for the current perturbation type. 
        force_accept_ratio_curr = force_accept_ratio[pert_type]
        if force_accept_ratio_curr == 0.0:
            accept = False
            force_accept = False
            return accept, force_accept
        else: 
            # Check for forced acceptance 
            force_accept = (rng_.uniform(0, 1) < force_accept_ratio_curr and 
                            self.data_misfit[ind] < self.max_misfit_force)
            # Make accept/reject decision.
            if force_accept:
                accept = True
            else:
                accept = False
            return accept, force_accept

    def calc_accept_ratio(self, ind): 
        """ 
        Proxy for Metropolis-Hastings acceptance ratio 
        using log likelihood and log petro prior and log geometrical prior. 
        """

        # Get the sum of terms. 
        term_sum = (self.log_likelihood_ratio[ind] + 
                    self.log_priorgeom_ratio[ind] + 
                    self.log_priorpetro_ratio[ind])

        # Get the log of accept ratio. 
        # Metropolis-Hastings: min(1, exp(log_ratio)). Here, take the log of this. 
        log_accept_ratio = min(0., term_sum)

        # Get accept ratio.
        if log_accept_ratio == 0.:
            self.accept_ratio[ind] = 1.
        elif np.abs(log_accept_ratio) > InversionMetrics.MAX_LOG_RATIO:  # To avoild calculating exponentials of large of very small values.
            self.accept_ratio[ind] = 0.
        else: 
            self.accept_ratio[ind] = np.exp(log_accept_ratio)
        return None
    
    def count_by_type(self, pert_type: int) -> int:
        """Count how many times a specific perturbation type was accepted."""
        # Handle both list and numpy array
        if isinstance(self.it_accepted_type, np.ndarray):
            return int(np.sum(self.it_accepted_type == pert_type))
        else:
            return self.it_accepted_type.count(pert_type)

    def get_iterations_for_type(self, pert_type: int) -> list:
        """Get all iterations where a specific perturbation type was accepted."""
        # Handle both list and numpy array
        if isinstance(self.it_accepted_type, np.ndarray):
            mask = self.it_accepted_type == pert_type
            return [int(it) for it in np.array(self.it_accepted_model)[mask]]
        else:
            return [int(it) for it, pt in zip(self.it_accepted_model, self.it_accepted_type) 
                    if pt == pert_type]

    def get_acceptance_summary(self, run_params) -> dict:
        """
        Get summary of accepted changes.
        
        Args:
            run_params: RunBaseParams instance for perturbation type names
        """
        summary = {'total_accepted': len(self.it_accepted_model), 'by_type': {}}
        
        for pert_type in range(5):  # 0-4 are valid types
            count = self.count_by_type(pert_type)
            iterations = self.get_iterations_for_type(pert_type)
            
            # Extract clean name from message (e.g., "pert_type = 1 -> Random..." -> "Random...")
            full_msg = run_params.perturbation_messages.get(pert_type, f"Unknown type {pert_type}")
            clean_name = full_msg.split("->")[-1].strip() if "->" in full_msg else full_msg
            
            summary['by_type'][clean_name] = {'pert_type': pert_type, 'count': count,'iterations': iterations}
        
        return summary