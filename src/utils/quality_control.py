"""
Quality Control Module.

This module provides exception classes and validation functions for
inversion and null space navigation operations.
Centralizes all quality control logic.
"""

import numpy as np
import inspect
from typing import Union, Any, Callable, Optional
import warnings

class CodeBug(Exception):
    """Base exception when a bug is detected by a check."""
    pass

# ============================================================================
# EXCEPTIONS - Domain-specific errors.
# These exceptions are not part of log_run. TODO unify with logging??
# ============================================================================
class ModelPerturbationError(Exception):
    """Base exception for pertubation operations."""
    pass

class ModelValueError(Exception):
    """Base exception for model check."""
    pass

class NucleationError(ModelPerturbationError):
    """Raised when birth/nucleation operations fail."""
    pass

class DataMisfitError(ModelPerturbationError):
    """Raised when data misfit exceeds acceptable bounds."""
    pass

class ParameterValidationError(ModelPerturbationError):
    """Raised when parameter validation fails."""
    pass

class BirthParameterValidationError(ModelPerturbationError):
    """Raised when model validation fails."""
    pass

def raise_perturbation_error(pert_type, i, log_run):
    pert_names = {0: "forced", 1: "geometric", 2: "petrophysical", 3: "birth", 4: "death"}
    pert_name = pert_names.get(pert_type, f"unknown type {pert_type}")
    
    msg = (f"Perturbation failed: {pert_name} (type {pert_type}) at iteration {i}\n"
           f"Possible causes:\n"
           f"  - Check output_*/m_curr_{i}.vts for model state\n"
           f"  - Review last 10 log entries for warnings\n"
           f"  - If birth/death: check n_units_total in metrics")
    
    log_run.error(msg)
    raise ModelPerturbationError(msg)

# ============================================================================
# WARNINGS.
# ============================================================================

class PetroOptimizationWarning(Warning):
    """Raised when optimization routines fail."""
    pass

class GeneralWarning(UserWarning):
    """Raised when data quality is questionable but within bounds."""
    pass

def issue_warning(message: str, category: type = UserWarning, 
                 stacklevel: int = 2, callback_printer: Optional[Callable] = None):
    """
    Issue a warning with optional logging.
    
    Parameters
    ----------
    message : str
        Warning message
    category : type, default=UserWarning
        Warning category class
    stacklevel : int, default=2
        Stack level for warning origin
    callback_printer : callable, optional
        Logger function to also print warning
    """
    warnings.warn(message, category=category, stacklevel=stacklevel)
    
    if callback_printer is not None:
        callback_printer(f"WARNING: {message}")


def validate_unit_count(mvars, petrovals, raise_error=False, callback_printer=None):
    """
    Validate consistency between signed distance dimensions and tracked units.
    
    Checks that the number of units in phi_curr (geometric representation)
    matches the number of tracked units in petrovals (petrophysical state).
    Inconsistencies indicate a bug in birth/death operations or state tracking.
    
    Parameters
    ----------
    mvars : ModelStateManager
        Model state with phi_curr signed distances
    petrovals : PetroStateManager
        Petrophysical state with tracked units
    raise_error : bool, default=False
        If True, raise ParameterValidationError on mismatch.
        If False, log error and return False.
    callback_printer : callable, optional
        Function for logging messages (e.g., log_run.error)
        
    Returns
    -------
    bool
        True if unit counts are consistent, False otherwise
        
    Raises
    ------
    ParameterValidationError
        If raise_error=True and validation fails
        
    Notes
    -----
    This validation is critical after birth and death operations to ensure
    the geometric representation (phi_curr) stays synchronized with the
    petrophysical tracking (petrovals).
    
    Common causes of failure:
    - Birth added unit to phi_curr but not to petrovals.insert_value()
    - Death removed unit from phi_curr but not petrovals.remove_value()
    - Manual modification of phi_curr without updating petrovals
    
    Examples
    --------
    >>> # After a birth operation
    >>> if not validate_unit_count(mvars, petrovals, callback_printer=log_run.error):
    ...     # Handle inconsistency
    
    >>> # With exception raising
    >>> validate_unit_count(mvars, petrovals, raise_error=True)  # Raises on error
    """
    # # Check if phi_curr is initialized
    # if mvars.phi_curr is None:
    #     error_msg = "Validation failed: mvars.phi_curr is not initialized (None)"
    #     if callback_printer:
    #         callback_printer(error_msg)
    #     if raise_error:
    #         raise ParameterValidationError(error_msg)
    #     return False
    
    # # Check if petrovals tracking is initialized
    # if petrovals._tracked_state is None:
    #     error_msg = "Validation failed: petrovals tracking is not initialized (None)"
    #     if callback_printer:
    #         callback_printer(error_msg)
    #     if raise_error:
    #         raise ParameterValidationError(error_msg)
    #     return False
    
    # Get unit counts from both sources
    n_units_phi = mvars.phi_curr.shape[0]
    n_units_petro = len(petrovals._tracked_state["items"])
    
    # Check for consistency
    if n_units_phi != n_units_petro:
        error_msg = (
            f"Unit count mismatch between geometry and tracking:\n"
            f"  - phi_curr has {n_units_phi} signed distance fields\n"
            f"  - petrovals has {n_units_petro} tracked units\n"
            f"  - Difference: {n_units_phi - n_units_petro}\n"
            f"This indicates a synchronization bug in birth/death operations."
        )
        
        # Get detailed breakdown for debugging
        if callback_printer:
            callback_printer(error_msg)
            callback_printer(f"phi_curr.shape: {mvars.phi_curr.shape}")
            callback_printer(f"petrovals.all_values: {petrovals.all_values}")
            
            # Count by origin
            n_original = sum(1 for item in petrovals._tracked_state["items"] if item[2] == 'orig')
            n_birthed = sum(1 for item in petrovals._tracked_state["items"] if item[2] == 'birth')
            callback_printer(f"  - Original units: {n_original}")
            callback_printer(f"  - Birthed units: {n_birthed}")
            
        if raise_error:
            raise ParameterValidationError(error_msg)
        
        return False
    
    # Validation passed
    return True


def validate_model_state_consistency(mvars, petrovals, check_values=True, 
                                     raise_error=False, callback_printer=None):
    """
    Comprehensive validation of model state consistency.
    
    Performs multiple consistency checks between geometric and petrophysical
    state representations. More thorough than validate_unit_count alone.
    
    Parameters
    ----------
    mvars : ModelStateManager
        Model state manager
    petrovals : PetroStateManager
        Petrophysical state manager
    check_values : bool, default=True
        If True, also validate that petrovals.all_values match model values
    raise_error : bool, default=False
        If True, raise exception on first failure
    callback_printer : callable, optional
        Logging function
        
    Returns
    -------
    bool
        True if all checks pass, False otherwise
        
    Raises
    ------
    ParameterValidationError
        If raise_error=True and any validation fails
        
    Notes
    -----
    Checks performed:
    1. Unit count consistency (phi_curr vs petrovals)
    2. Value consistency (petrovals.all_values vs model values)
    3. State initialization
    
    Examples
    --------
    >>> # After complex operation
    >>> if not validate_model_state_consistency(mvars, petrovals, 
    ...                                          callback_printer=log_run.error):
    ...     log_run.error("Model state inconsistent - reverting")
    ...     mvars.revert_to_prev()
    ...     petrovals.revert_to_prev()
    """
    all_passed = True
    
    # Check 1: Unit count consistency
    if not validate_unit_count(mvars, petrovals, raise_error=False, 
                              callback_printer=callback_printer):
        all_passed = False
        if raise_error:
            raise ParameterValidationError("Unit count validation failed")
        else: 
            if callback_printer is not None: 
                callback_printer(f"Unit count validation failed")
    
    # Check 2: Petrophysical value consistency
    if check_values and mvars.m_curr is not None:
        if not petro_consistency_check(petrovals.all_values, mvars.m_curr, 
                                      callback_printer=callback_printer):
            all_passed = False
            if raise_error:
                raise ParameterValidationError("Petrophysical value consistency check failed")
            else: 
                if callback_printer is not None: 
                    callback_printer(f"Unit count validation failed")
    
    # # Check 3: Basic initialization
    # if mvars.m_curr is None:
    #     error_msg = "mvars.m_curr is None - model not initialized"
    #     if callback_printer:
    #         callback_printer(error_msg)
    #     all_passed = False
    #     if raise_error:
    #         raise ParameterValidationError(error_msg)
    
    # if mvars.phi_curr is None:
    #     error_msg = "mvars.phi_curr is None - signed distances not initialized"
    #     if callback_printer:
    #         callback_printer(error_msg)
    #     all_passed = False
    #     if raise_error:
    #         raise ParameterValidationError(error_msg)
    
    return all_passed


def birth_pass_sanity(metrics, mvars, i, petrovals, max_misfit, callback_printer=None):
    """
    Check misfit and petrophysical constraints after birth, revert model and save if check fails.

    Parameters:
        metrics: Object with data_misfit.
        mvars: Object with m_curr (model array).
        i (int): Current iteration index.
        petrovals: Object with all_values used for petro check.
        unique_vals: Unique values from model.
        counts: Occurrence counts of each value.
        max_misfit (float): Maximum allowed misfit value.
        callback_printer (callable): Logger or print function.
    
    Returns:
        bool: True if pass sanity check, False otherwise.
    """

    pass_misfit = sanity_check_misfit(metrics.data_misfit[i], max_value=max_misfit)
    pass_petrop = petro_consistency_check(petrovals.all_values, mvars.m_curr, callback_printer=callback_printer)

    if not pass_misfit or not pass_petrop:
        if not pass_petrop and callback_printer:
            callback_printer(f"Iteration {i}: Petro consistency check 4 not passed: Birth killed a unit. REJECT.")

        # Revert model and metrics
        mvars.m_curr.fill(-9999.)
        metrics.data_misfit[i] = metrics.data_misfit[i - 1]
        return False
    else: 
        return True


# TODO should this function be elsewhere?
def _get_petro_values(model_curr, return_counts_=False):
    return np.unique(model_curr, return_counts=return_counts_)
    

def validate_iteration_state(model_curr, petrovals_all_values, data_misfit, 
                             log_run, debug=True, max_misfit=50., context=""):
    """
    QC for some key values. 
    
    Args:
        model_curr: Current model array to check
        petrovals_all_values: Array of petrophysical values from petrovals object
        data_misfit: Current data misfit value to check
        log_run: Logger for diagnostic messages
        debug: Whether to perform validation (if False, always returns True)
        context: Additional context string for error messages
        
    Returns:
        bool: True if validation passes, False if validation fails
    """
    if not debug:
        return True
    
    else: 
        # Check misfit
        if not sanity_check_misfit(data_misfit, max_value=max_misfit, context=context, log_run=log_run):
            return False
        
        # Check petro consistency.  
        if not petro_consistency_check(petrovals_all_values, model_curr, callback_printer=log_run.error):
            return False
        
        return True


def sanity_check_misfit(misfit_val: Union[int, float, np.number], 
                        max_value: float = 50., 
                        context: str = "", log_run=None) -> bool:
    """
    Sanity check for data misfit values.
    
    Args:
        misfit_val: Misfit value to validate
        max_value: Maximum acceptable misfit value
        context: Additional context for error messages
        
    Returns:
        bool: True if validation passes.
        
    Raises:
        DataMisfitError: If validation fails
    """
    e = None
    # Type validation
    if not isinstance(misfit_val, (int, float, np.number)):
        e = DataMisfitError(
            f"Invalid misfit type: got {type(misfit_val).__name__}")

    # NaN validation
    if np.isnan(misfit_val):
        e = DataMisfitError("Misfit is NaN")
    
    # Infinite validation
    if np.isinf(misfit_val):
        e = DataMisfitError(f"Misfit is infinite {context}")

    # Negative validation
    if misfit_val < 0:
        e = DataMisfitError(f"Negative misfit value: {misfit_val} {context}")

    # Maximum value validation with caller location
    if misfit_val > max_value:
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        e = DataMisfitError(
            f"Misfit {misfit_val} exceeds maximum {max_value} "
            f"at {filename}:{lineno} {context}"
        )

    if e is not None: 
        if log_run:
            log_run.log_exception(e, context)
        else:
            raise e
    else:
        return True


def pass_sanity_check_mask(mask: np.ndarray, 
                          unit_to_perturb: Any, 
                          perturbing_unit: Any, 
                          petrovals_pert: Any, 
                          petrovals_obj: Any, 
                          callback_printer: Callable[[str], None]) -> bool:
    """
    Validate mask for perturbation operations with PetroQuantities integration.
    
    This function checks that the mask allows perturbations and that the
    required petrophysical values exist in the current model. Maintains
    compatibility with the integrated PetroQuantities class.
    
    Args:
        mask: Boolean mask array defining where perturbations are blocked
        unit_to_perturb: Value of the unit being targeted for perturbation
        perturbing_unit: Value of the unit doing the perturbing
        petrovals_pert: Perturbation values being applied
        petrovals_obj: PetroQuantities object containing tracked state
        callback_printer: Function for logging diagnostic messages
        
    Returns:
        bool: False if mask blocks all perturbations (validation fails),
              True if mask allows perturbations (validation passes)
              
    Note:
        This function logs detailed diagnostics but doesn't raise exceptions,
        allowing the caller to decide how to handle mask validation failures.
    """
    # Check if mask blocks all locations (np.all(mask) means all locations are blocked)
    if np.all(mask):
        # Log detailed diagnostic information
        callback_printer(f"unit_to_perturb: {unit_to_perturb}")
        callback_printer(f"petrovals.pert: {petrovals_pert}")
        callback_printer(f"perturbing_unit: {perturbing_unit}")
        
        # Log petrophysical object state for debugging
        if hasattr(petrovals_obj, 'tracked_state') and petrovals_obj.tracked_state:
            callback_printer(f"petro_dict: {petrovals_obj.tracked_state}")
            callback_printer(f"petro_dict['mapping']: {petrovals_obj.tracked_state['mapping']}")
            
        if hasattr(petrovals_obj, 'all_values'):
            callback_printer(f"petro_dict['arr']: {petrovals_obj.all_values}")
            
        callback_printer(
            "Error: petrophysical values for the mask not found in the current model. "
            "Stopping execution!!! \n "
        )
        
        return False
    
    return True


def petro_consistency_check(petro_array: np.ndarray, 
                            model_to_check: np.ndarray,
                            callback_printer: Optional[Callable] = None) -> bool:
    """
    Check consistency between petrophysical arrays and model values.
    
    Args:
        petro_array: Array of petrophysical values from dictionary
        unique_vals: Unique values from model
        counts: Count of each unique value
        callback_printer: Function for logging/printing messages
        
    Returns:
        bool: True if arrays are consistent
    """

    unique_vals, counts = _get_petro_values(model_to_check, return_counts_=True)

    if not np.array_equal(petro_array, unique_vals):
        callback_printer(f"Petrophysical values: {petro_array}")
        callback_printer(f"counts: {counts}")
        # Log the problem
        if callback_printer:
            callback_printer("The two petrophysical arrays are different. This is a problem!")
            callback_printer(f"petro_array = {petro_array}")
            callback_printer(f"unique_vals = {unique_vals}")

        # Print differences for different failure modes
        if len(petro_array) != len(unique_vals):
            if callback_printer:
                callback_printer(f"len(petro_array) = {len(petro_array)}")
                callback_printer(f"len(unique_vals) = {len(unique_vals)}")

                diff = np.sum(petro_array) - np.sum(unique_vals)
                callback_printer(f"np.sum(petro_array) - np.sum(unique_vals) = {diff}")
            return False
            
        # Element-by-element comparison
        diff_indices = np.where(petro_array != unique_vals)[0]
        for ii in diff_indices:
            if callback_printer:
                callback_printer(f"Index {ii}: petro_dict['arr'][{ii}] = {petro_array[ii]}, unique_vals = {unique_vals[ii]}")
                callback_printer(' ~~~~ Exiting main loop because values from petro_dict and unique vals from m_curr differ!! ~~~~\n')
        return False
    
    return True