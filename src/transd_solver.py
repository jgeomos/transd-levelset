"""
Solver for trans-dimensional inversion and model space exploration.

This module implements the core MCMC sampling algorithm for exploring the
space of geophysical inverse problems focussing on a geometrical approach, 
trans-dimensional sampling with birth/death of units and several
perturbation strategies (geometrical, petrophysical, forced).

Main Algorithm
--------------
The solver samples models to:
1. Propose changes to model geometry (perturbations of signed distances)
2. Propose changes to petrophysical properties (density, susceptibility)
3. Add new units (birth moves)
4. Remove existing units (death moves)
5. Accept/reject proposals using a (non-revertible) Metropolis-Hastings approach 

Key Functions
-------------
pertub_scalar_fields : function
    Main MCMC sampling loop that orchestrates all perturbation types
_apply_forced_perturbation : function
    Apply user-defined model changes with forced acceptance
_apply_geometrical_perturbation : function
    Perturb unit geometries via level set modifications
_apply_petrophysical_perturbation : function
    Modify petrophysical property values
_apply_birth_perturbation : function
    Nucleate new geological units at high-gradient (of data cost function) locations
_apply_death : function
    Remove existing units from the model

Perturbation Types
------------------
Type 0 : Forced perturbation
    Apply pre-defined model changes (e.g., imposed structure that grows or petrophysical value evolution)
Type 1 : Geometric perturbation
    Modify unit boundaries using correlated random fields
Type 2 : Petrophysical perturbation
    Change density/property values within units
Type 3 : Birth
    Add new geological units (increases dimensionality)
Type 4 : Death
    Remove units (decreases dimensionality)

Algorithm Flow
--------------
For each iteration:
    1. Select perturbation type (randomly or deterministically)
    2. Propose model change
    3. Calculate forward data and misfit
    4. Evaluate acceptance probability (likelihood × prior)
    5. Accept or reject based on Metropolis-Hastings criterion
    6. Update state if accepted, revert if rejected
    7. Record metrics

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

import numpy as np
from importlib import reload
import warnings

import src.utils.transd_utils as tu
import src.forward_solver.forward_calculation as fcu
import src.utils.quality_control as qc
import src.random_gen.anisotropic_noise as an
import src.random_gen.scalar_sampling as st
import src.input_output.output_manager as om

reload(fcu)
reload(tu)
reload(qc)
reload(an)
reload(om)
reload(st)

# TODO: a future development could be to create a module with array operations e.g array_utils.py,
#  or, more focussed, pure numpy operations, e.g. array_ops.py.

# Use the fact that the residual vector of a model can be used to avoid re-calculating the fwd

DEBUG = True  # set to False in production
MAX_MISFIT = 50
# DTYPE_DEFAULT = np.float32  # For use throughout the code


def _generate_random_field(config):
    """
    Generate (normalised) random noise with a given spectrum with some blobs.
    
    This function:
    1. Keeps generated values at indices provided in 'mask', assigns zero elsewhere.
    2. Multiplies the values by weights given in 'weights'.
    
    Args:
        config: Configuration object containing all parameters
        
    Returns:
        array: Values simulating random noise with specified properties
    """
    
    # Sample values for the perturbations
    # factor_spectrum = config.rng.uniform(config.factor_spectrum_min, config.factor_spectrum_max)
    # amplitude_pert = config.rng.uniform(config.amplitude_pert_min, config.amplitude_pert_max)
    factor_spectrum = st.sample_uniform(config.factor_spectrum_min, config.factor_spectrum_max, config.rng)
    amplitude_pert = st.sample_uniform(config.amplitude_pert_min, config.amplitude_pert_max, config.rng)

    # Generate a 3D volume of random noise
    random_pert = an.create_anisotropic_noise_field(
                        dimensions=config.dimensions[1:], 
                        factor_spectrum=factor_spectrum, 
                        rng=config.rng,
                        correlation_length=config.correlation_length, 
                        corr_zx=config.corr_zx, 
                        corr_zy=config.corr_zy, 
                        corr_xy=config.corr_xy, 
                        rotation_angle=config.rotation_angle,
                        cell_sizes=config.cell_sizes, 
                        rotation_matrix=config.rotation_matrix, 
                        fft_class=config.fft_class
                        )

    # Normalize to have histogram of values centered on 0
    non_zeros = np.where(random_pert != 0)
    if config.normalise:
        an.normalize_to_symmetric_range(random_pert, non_zeros, target_range=amplitude_pert)

    # # Make it zero-mean
    # random_pert[non_zeros] -= np.mean(random_pert[non_zeros])
    # # Set the amplitude
    # random_pert[non_zeros] *= amplitude_pert

    # Apply mask if provided
    if config.mask is not None:
        random_pert = random_pert.flatten() 
        random_pert *= config.mask.flatten()
        random_pert = random_pert.reshape(config.dimensions[1:])

    # Sanity check
    if np.max(random_pert) <= 0.:
        raise qc.ModelPerturbationError("The max value of pert should be > 0!")
    
    return random_pert


def _birth_update_model_and_misfit(pert_dens_nucl, birth_params, model_class, sensit, geophy_data):
    """
    Optimize density values in nucleation region and evaluate resulting model.
    """
    #TODO split this function in 2, one for update and one for misfit?

    new_value_birth = tu.get_birth_value(model_class, pert_dens_nucl, birth_params)

    # Calculate resid vect and misfit for proposed model.
    # Calculate forward data and misfit for trial model
    original_data_calc = geophy_data.data_calc
    
    # Calculate new data and get resid vect and misfit.
    fcu.calc_geophy_data(geophy_data, sensit, model_class.m_tmp)
    resid_vect_tmp, misfit_data_tmp = fcu.calc_data_rms(geophy_data)
    
    # Restore original data_calc for geophy_data
    geophy_data.data_calc = original_data_calc

    return pert_dens_nucl, new_value_birth, misfit_data_tmp, resid_vect_tmp


def _create_nucleation_mask(birth_params, model_mcurr, grad_cost):
    """
    Create mask defining where nucleation can occur.
    0 = excluded, 1 = potential nucleation site.
    """
    # Start with zeros (all excluded)
    mask = np.zeros(model_mcurr.shape, dtype=np.float32)
    
    # Set high-gradient areas as candidates
    mask[birth_params.max_grad_ind] = 1.0
    
    # Remove excluded indices (if they overlap, we want them excluded)
    if birth_params.exclude_indices is not None:
        mask[birth_params.exclude_indices] = 0.0
    
    return define_nucl_mask(birth_params, mask, model_mcurr, grad_cost)


def define_nucl_mask(birth_params, mask, model_curr, gradient_cost):
    """
    Defines masks that control the locations for the birth of rock units. There are 3 cases:
    1. nucl_geol = 0. No domains: geometry of units/group of units not accounted for, the whole volume can be used.
    2. nucl_geol = 1. Domains follow units put together in groups.
    3. nucl_geol = 2. Domains follow all rock units.
    """

    # ---- Do the masks for birth of units depending on the different scenarios.
    # Case where no geology is consider
    if birth_params.nucl_geol == 0:
        # identify which rock unit in the subvolume has the highest data misfit/sensit and create mask
        mask, birth_params.ind_nucl = tu.nucl_calc_mask(mask=mask, model=np.ones_like(model_curr), domains=np.ones(1),
                                             grad_cost=gradient_cost)
    elif birth_params.nucl_geol == 1:
        if birth_params.domains_births is None:
            raise qc.NucleationError(
                f"The current domains or mask prevents the creation of any domain where nucleation can occur! Cannot do a birth!"
                f"domains_births={birth_params.domains_births}")
        
        groups_indices = np.unique(birth_params.domains_births)
        # identify which rock unit in the subvolume has the highest data misfit/sensit and create mask
        mask, birth_params.ind_nucl = tu.nucl_calc_mask(mask=mask, model=birth_params.domains_births, domains=groups_indices,
                                             grad_cost=gradient_cost)

    elif birth_params.nucl_geol == 2:
        petro_values = tu.get_petro_values(model_curr)
        # identify which rock unit in the subvolume has the highest data misfit/sensit and create mask
        mask, birth_params.ind_nucl = tu.nucl_calc_mask(mask=mask, model=model_curr, domains=petro_values,
                                             grad_cost=gradient_cost)

    else:
        raise qc.ParameterValidationError(f"Invalid domain definition: nucl_geol={birth_params.nucl_geol}: expected 0, 1, or 2.")

    return mask

def _try_nucleation_in_region(birth_params, grad_cost, model_class, sensit, geophy_data, resid_vect):
    """
    Try nucleation in a specific connected region.
    Depending on, birth_params.nucl_geol the connected region will be cropped to conform to units, groups of units and be unconstrained.

    Returns
        relative_value_added_to_model, new_petro_value, misfit_of_new_mdel, resid_of_new_mdel
    """

    # Generate mask controlling where birth can occur based on the provided gradient of cost and current model. 
    model_mask = _create_nucleation_mask(birth_params, model_class.m_curr, grad_cost)

    # Calculate the data for the non-maked region. 
    resp_masked = fcu.calc_fwd(sensit, model_mask, unit_conv=geophy_data.unit_conv)

    # Calculate optimal petro perturbation: relative perturbation, a value added to that already in the model in masked area.
    petro_opti = tu.calc_petro_opti(resp_masked, resid_vect, birth_params.bounds)
    if not petro_opti.success:
        warnings.warn(
            f"Bounded least squares optimization failed: {petro_opti.message}.\n"
            f"Iterations: {petro_opti.nit},\n"
            f"Optimality: {petro_opti.optimality:.2e},\n"
            f"Matrix shape: {resp_masked.shape},\n"
            f"Residuals shape: {resp_masked.shape}\n",
            f"  Bounds: {birth_params.bounds}\n"
            f"  Try: Increase n_line_search or adjust bounds in BirthConfig????",
            category=qc.PetroOptimizationWarning
        )
    pert_dens_nucl = petro_opti.x
    

    return _birth_update_model_and_misfit(pert_dens_nucl, birth_params, model_class, sensit, geophy_data)
    

def _birth_deterministic(birth_params, misfit_data_curr, sensit, geophy_data, resid_vect, model_class, log_run=None):
    """
    Deterministically birth a new unit at gradient-optimized location.
    
    Uses data misfit gradient to identify optimal nucleation site and
    petrophysical value (rather than stochastic sampling). Evaluates
    high-gradient regions and optimizes for best data fit.
    
    Parameters
    ----------
    mvars : ModelStateManager
    petrovals : PetroStateManager
    birth_params : BirthConfig
    geophy_data : GeophysicalData
    sensit : Sensitivity
    log_run : Logger
        
    Returns
    -------
    success : bool
        True if operation completed
    birth_occurred : bool  
        True if new unit was birthed
    optimal_value : float or None
        Petrophysical value of birthed unit
    """
    # TODO Cover the case a proposed birth has a density exactly equal to another unit. -> This becomes another perturbation entirely!

    birth_params.set_gradient_thresholds()

    mod_dens_best = model_class.m_curr.copy()  # TODO replace mod_dens_best by model_class.m_tmp, for in-place calc?
    resid_vect_tmp = resid_vect.copy()
    resid_vect_best = resid_vect.copy()
    new_values_birth_best = None
    birth_happend = False
    if model_class.m_tmp is None: 
        model_class.m_tmp = model_class.m_curr.copy() 
    else: 
        np.copyto(model_class.m_tmp, model_class.m_curr)
    
    # Calculate the gradient of the data misfit.
    grad_cost = fcu.calc_grad_misfit(sensit, geophy_data)

    for j in range(birth_params.n_line_search):

        # Get connected components the different groups of cells gradient superior to a given treshold. 
        mask_grad_labels, n_blobs = tu.calc_gradient_connected_regions(grad_cost, birth_params, j)

        # Test birth in the different connected regions of gradient above threshold.
        for jj in range(1, n_blobs + 1): 
            
            # TODO can the mask for nucleation be calculated once and for all? 
            
            # extract values corresponding to min and max gradient of data misfit for the current region.
            birth_params.max_grad_ind = np.where(mask_grad_labels == jj)

            # Check condition on minimum number of cells to make a birth.
            if np.shape(birth_params.max_grad_ind)[1] > birth_params.min_cells_num:

                # This function will create masks to conform to geology and account for constraints when proposing births. 
                pert_dens_nucl, new_values_birth, misfit_data_tmp, resid_vect_tmp = _try_nucleation_in_region(
                                birth_params, grad_cost, model_class, sensit, geophy_data, resid_vect)

                if misfit_data_tmp < misfit_data_curr:
                    misfit_data_curr = misfit_data_tmp
                    np.copyto(mod_dens_best, model_class.m_tmp) 
                    new_values_birth_best = new_values_birth
                    np.copyto(resid_vect_best, resid_vect_tmp)
                    birth_happend = True

                    if log_run is not None: 
                        log_run.info(f'\t Data misfit new best: {misfit_data_tmp}')
                        log_run.info(f'\t Birth value for new best: {new_values_birth}')
                        log_run.info(f'\t Proposed perturbation of density: {pert_dens_nucl}')

    if not birth_happend:
        if log_run is not None: 
            log_run.info("\t No better model could be found from nucleation for birth!")

    return mod_dens_best, misfit_data_curr, new_values_birth_best, resid_vect_best, birth_happend


def perturb_petro(petrovals, index_unit_pert_, rng_seed, max_attempts=100):
    """ Perturbation of rock unit densities 
    Args:
            petrovals: PetroStateManager object
            index_unit_pert: Index of unit to perturb
            rng_seed: Random number generator
            max_attempts: Maximum attempts before giving up
    
    Returns:
        float: New unique density value
        
    Raises:
        ParameterValidationError: If the provide distribution is not of a supported type.
        NucleationError: If the resulting perturbation is 0.0 or results in growing/shrinking another unit (rare case where a proposed value 
        randomly lands on another value existing already in the model).
    """
    
    base_value = petrovals.all_values[index_unit_pert_]
    attempted = set(petrovals.all_values) 

    petro_pert = 0.0
    attempts = 0

    while petro_pert == 0.0 and attempts < max_attempts:
        if petrovals.distro_type == 'uniform':  
            petro_pert = st.sample_uniform(- petrovals.std_petro, + petrovals.std_petro, rng_seed, round_perc=0.005)
        elif petrovals.distro_type == 'gaussian':
            petro_pert = st.sample_gaussian(mu=0.0, sigma=petrovals.std_petro, rng_s=rng_seed, round_perc=0.005)
        else: 
            raise qc.ParameterValidationError(f"Invalid distribution type: {petrovals.distro_type}. Must be 'uniform' or 'gaussian'.")
        attempts += 1

    if petro_pert == 0.0:
        raise qc.NucleationError("Unable to generate non-zero perturbation")
    
    new_value = base_value + petro_pert

    # Ensure the new value is not already in all_values
    if new_value not in attempted:
        return new_value
    else:
        raise qc.NucleationError(f"Could not generate a perturbation leading to a new unit! Attempted: {len(attempted)} values")


def _apply_petrophysical_perturbation(petrovals, mvars, metrics, shpars, pert_index_force, rng_main, 
                                     geophy_data, sensit, log_run, i):
    """
    Apply petrophysical perturbation (pert_type == 2).

    Perturb petrophysical property values of existing units.
    
    Modifies density or other property values without changing unit geometry.
    Samples new values from prior distribution and evaluates data fit.
    Used to refine property estimates while maintaining spatial structure.
    
    Parameters
    ----------
    petrovals : PetroStateManager
        Petrophysical state with tracked properties
    mvars : ModelStateManager
        Model state with current geometry
    metrics : InversionMetrics
        Metrics tracker for recording changes
    shpars : ShapeParameters
        Shape and sampling parameters
    pert_index_force : int
        Index of unit to perturb for forced perturbation.
    rng_main : numpy.random.Generator
        Random number generator
    geophy_data : GeophysicalData
        Observed geophysical data
    sensit : Sensitivity
        Sensitivity matrix for forward modeling
    log_run : Logger
        Logger for operation details
    i : int
        Current iteration number
        
    Returns
    -------
    success : bool
        True if perturbation completed successfully
    index_tmp : int
        Index of perturbed unit (same as pert_index_force)
        
    Notes
    -----
    Geometry (signed distances) remains unchanged - only property values
    are modified. Uses petrovals.modify_value_by_index() for tracking.
    """
    # TODO shpars.indices_unit_pert indices should be changed dynamically cause the indices of units change with births.
    index_tmp = None

    # Perturbation of density of unit selected at random.
    if shpars.force_pert_type is not None:
        index_unit_pert_curr = shpars.ind_unit_force
        while shpars.ind_unit_force == index_unit_pert_curr:
            log_run.info("Making a different choice of unit to pert (petrophy pert)")
            index_unit_pert_curr = st.select_random_index(rng_main, indices=shpars.indices_unit_pert)
            index_tmp = index_unit_pert_curr
            old_dens = petrovals.get_value_by_orig_index(index_unit_pert_curr)['val']
    else:
        index_unit_pert_curr = st.select_random_index(rng_main, indices=shpars.indices_unit_pert)
        index_tmp = index_unit_pert_curr.copy()
        old_dens = petrovals.get_value_by_orig_index(index_unit_pert_curr)['val']

    log_run.info(f"Perturbation of DENSITY for unit (original index): {index_unit_pert_curr}, with OLD density = {old_dens}")

    # Get index in updated petro dict. 
    index_unit_pert_update = petrovals.get_value_by_orig_index(index_unit_pert_curr)['ind']

    petro_pert_curr = perturb_petro(petrovals, index_unit_pert_update, rng_main)
    log_run.info(f"Perturbation of DENSITY for unit (current index) : {index_unit_pert_update}, with NEW density =  {petro_pert_curr}")

    petrovals.modify_value_by_index(ind=index_unit_pert_update, new_value=petro_pert_curr)
    mvars.m_curr = tu.get_density_model(petrovals.all_values, mvars.phi_curr)

    fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)
    metrics.calc_data_rms_misfit(geophy_data, ind=i)

    # Sanity checks.
    if not qc.validate_iteration_state(mvars.m_curr, petrovals.all_values, metrics.data_misfit[i], log_run, debug=DEBUG, max_misfit=MAX_MISFIT):
        return False, index_tmp
        
    if petro_pert_curr == old_dens:
        log_run.info("SAME DENSITY FOR BOTH OLD AND NEW! Should NOT BE THE CASE. Check how perturbation is done!!! EXITING")
        return False, index_tmp

    # Make sure values of petrovals.pert are up to date (useful for pert_type=1 in mvars.m_curr[mvars.phi_aux[0] > 0] = petrovals.pert)
    if shpars.force_pert_type in ('petrophy_increase', 'petrophy_decrease'):
        if (index_tmp == pert_index_force):
            petrovals.pert = petrovals.get_value_by_orig_index(pert_index_force)['val']  
            log_run.info(f"petrovals.pert = {petrovals.pert}")
    
    return True, index_tmp


def _apply_forced_perturbation(petrovals, mvars, metrics, shpars, spars, gpars, phipert_config, rng_main, 
                               geophy_data, sensit, log_run, i):
    """
    Apply forced perturbation (pert_type == 0).
    
    Returns:
        bool: success - whether perturbation was applied successfully
    """
    force_pert_type = shpars.force_pert_type
    use_dynamic_mask = shpars.use_dynamic_mask
    ind_unit_ref = shpars.ind_unit_ref
    
    mvars.index_unit_pert = shpars.ind_unit_force

    # Define dynamic masks to prevent certain units to be affected by perturbations. 
    if use_dynamic_mask:
        # Original unit.
        unit_to_perturb = petrovals.get_value_by_orig_index(mvars.index_unit_pert)['val']
        # Unit that grows inside Original unit.
        perturbing_unit = petrovals.get_value_by_orig_index(ind_unit_ref)['val']
        # Get mask defining locations where changes cannot occur. 
        mask_force_pert = ~np.isin(mvars.m_curr, [unit_to_perturb, perturbing_unit])
        
        # Sanity check.
        if not qc.pass_sanity_check_mask(mask_force_pert, unit_to_perturb, perturbing_unit, 
                                    petrovals.pert, petrovals, log_run.error):
            return False
    else: 
        mask_force_pert = None
    
    jj = 0
    while np.array_equal(mvars.m_prev, mvars.m_curr):

        if force_pert_type in ('petrophy_increase', 'petrophy_decrease'):

            index_unit_pert_curr = mvars.index_unit_pert
            index_tmp = index_unit_pert_curr
            old_dens = petrovals.get_value_by_orig_index(index_unit_pert_curr)['val']

            log_run.info(f"Perturbation of DENSITY for unit (original index): {index_tmp}, with OLD density = {old_dens}")

            # Get index in updated petro dict. 
            index_unit_pert_update = petrovals.get_value_by_orig_index(index_unit_pert_curr)['ind']
            
            match force_pert_type:
                case 'petrophy_increase':
                    petro_pert_curr = old_dens + st.sample_uniform(min_value=0.1, max_value=5., rng_s=rng_main)
                case 'petrophy_decrease':
                    petro_pert_curr = old_dens - st.sample_uniform(min_value=0.1, max_value=5., rng_s=rng_main)

            log_run.info(f"Propose FORCED PETRO. perturbation for unit (current index) : {index_unit_pert_update}, with NEW density =  {petro_pert_curr}")

            # Cases the increase (decrease) in petro makes it more (less) than unit above (below, in terms of petrophy prop.). 
            if (petro_pert_curr >= petrovals.all_values[index_unit_pert_update+1]) or (petro_pert_curr <= petrovals.all_values[index_unit_pert_update-1]):
                log_run.info(f"The petrophy value of the perturbed unit has reached that of the neighbouring unit's: stopping.")
                return False

            petrovals.modify_value_by_index(ind=index_unit_pert_update, new_value=petro_pert_curr)
            mvars.m_curr = tu.get_density_model(petrovals.all_values, mvars.phi_curr)

            fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)
            metrics.calc_data_rms_misfit(geophy_data, ind=i)

        elif force_pert_type == 'geometry':

            log_run.info(f"Force modifying rock {mvars.index_unit_pert} with density {petrovals.all_values[mvars.index_unit_pert]}")
            # Generate random field for pert of signdist of forced geometry anomaly.
            pert = _generate_random_field(phipert_config)
            
            # Keep only positive values to prevent anomaly from shrinking. 
            pert[pert < 0.] = 0. 
            pert *= mvars.loaded_mask if shpars.use_loaded_mask else 1
            
            # Constrain the perturbation so it does not grow except within certain units. 
            pert[mask_force_pert] = 0.

            # Set some cells as un-modifyable.
            mvars.phi_aux[0, mvars.phi_aux[0] > 0] = + np.inf
            # Apply perturbation to signdist. 
            mvars.phi_aux[0] += pert
            
            # Calc the density model of updated anomaly. 
            indices_force_mod = np.where(np.squeeze(mvars.phi_aux[0]) > 0)
            mvars.mod_aux[indices_force_mod] = petrovals.all_values[mvars.index_unit_pert]

            # Re-init signdist for updated anomaly. 
            mvars.phi_curr[mvars.index_unit_pert, mvars.phi_aux[0] > 0] = + np.inf
            # Update model.
            mvars.m_curr = tu.get_density_model(petrovals.all_values, mvars.phi_curr)

            # Calculate updated signed distances to calculate prior model cost. 
            mvars.phi_curr = tu.calc_signed_distances_opti(mvars.m_curr, mvars.m_prev, mvars.phi_prev, petrovals.all_values, 
                                                        cell_size=[1, 1, 1], narrow=True, log_run=log_run)

        jj += 1
        if jj >= 25:
            # Save files for later debugging.
            om.save_random_field_perturbation_outputs(i, mvars, pert, mask_force_pert, gpars, spars, log_run)
            log_run.info("Exiting main loop: geometrical perturbation using auxiliary model does not change the model. Saved relevant output. ")
            return False
        
        if np.array_equal(mvars.m_prev, mvars.m_curr):
            mvars.revert_to_prev()
            log_run.info("No modification enforcing delta_m: try another again! ")
    
    return True


def _apply_geometrical_perturbation(petrovals, mvars, metrics, shpars, spars, phipert_config, gpars, rng_main, 
                                    geophy_data, sensit, log_run, i):
    """
    Perturb unit geometry via random level set field modifications.
    
    Applies spatially correlated random perturbations to signed distance
    functions, changing unit shapes and interfaces. Property values remain
    constant. Used for exploring geometric uncertainty in model structure.
    
    Parameters
    ----------
    petrovals : PetroStateManager
        Petrophysical state manager
    mvars : ModelStateManager
        Model state with signed distances to perturb
    metrics : InversionMetrics
        Metrics tracker
    shpars : ShapeParameters
        Shape and sampling parameters
    spars : SaveParameters
        Save/output parameters
    phipert_config : PhiPertConfig
        Configuration for level set perturbations (correlation length, etc.)
    gpars : GeometryParameters
        Grid geometry parameters
    rng_main : numpy.random.Generator
        Random number generator
    geophy_data : GeophysicalData
        Observed geophysical data
    sensit : Sensitivity
        Sensitivity matrix
    log_run : Logger
        Logger for operation details
    i : int
        Current iteration number
        
    Returns
    -------
    success : bool
        True if perturbation completed successfully
        
    Notes
    -----
    Generates anisotropic random field and adds to phi_curr, then 
    regenerates m_curr from modified signed distances. Preserves
    petrophysical values while changing spatial distribution.
    """

    path_output = spars.path_output
    filename_model_save_rt = spars.filename_model_save_rt
    save_plots = spars.save_plots

    # Parameter controlling interface thickness.
    tau = 0.51 
    
    while np.array_equal(mvars.m_prev, mvars.m_curr):
        # Generate perturbation. 
        pert_phi = _generate_random_field(phipert_config)  

        # Get a rock unit at random and apply perturbation to it.
        # print(shpars.indices_unit_pert)
        # print(type(shpars.indices_unit_pert))
        
        index_unit_pert_curr = st.select_random_index(rng_main_=rng_main, indices=shpars.indices_unit_pert)
        
        index_unit_pert_update = petrovals.get_value_by_orig_index(index_unit_pert_curr)['ind']
        log_run.info(f"Perturbation of GEOMETRY for unit (current index) : {index_unit_pert_update}")

        if shpars.use_loaded_mask:
            pert_phi[mvars.loaded_mask == 0] = 0

        # Get interfaces of the current unit (value of 0.51: only the cells exactly at the interface).
        interface = tu.get_interface(mvars.phi_curr, index_unit_pert_update, tau)

        # Perturb the signed distances. 
        mvars.phi_curr[index_unit_pert_update, :, :, :] += pert_phi * interface
        # Get the perturbed petrophysical model.
        mvars.m_curr = tu.get_density_model(petrovals.all_values, mvars.phi_curr)

        fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)
        metrics.calc_data_rms_misfit(geophy_data, ind=i)

        if shpars.force_pert_type in ('petrophy_increase', 'petrophy_decrease'):
            log_run.info(f"petrovals.pert = {petrovals.pert}")
            mvars.m_curr[mvars.phi_aux[0] > 0] = petrovals.pert  # Make sure the values of forced pert remain the same. 

        if not qc.validate_iteration_state(mvars.m_curr, petrovals.all_values, metrics.data_misfit[i], log_run,  debug=DEBUG):
            om.save_model_to_vtk(mvars.m_curr.flatten(), gpars, 
                                filename=path_output + '/' + filename_model_save_rt + str(i), 
                                save=save_plots, log_run=log_run)
            return False

    # Calculate updated signed distances to calculate prior model cost.
    mvars.phi_curr = tu.calc_signed_distances_opti(mvars.m_curr, mvars.m_prev, mvars.phi_prev, petrovals.all_values, 
                                                   cell_size=gpars.cell_sizes, narrow=True, log_run=log_run)
    return True


def _apply_birth(birth_params, petrovals, mvars, metrics, gpars, spars, resid_vect, geophy_data, sensit, log_run, i):
    """
    Birth a new geological unit via gradient-guided nucleation.
    
    Analyzes data misfit gradient to identify high-gradient regions,
    evaluates connected regions for nucleation, optimizes petrophysical
    value, and adds new unit to model. Increases model dimensionality
    in trans-dimensional sampling framework.
    
    Parameters
    ----------
    birth_params : BirthConfig
        Birth/nucleation configuration parameters
    petrovals : PetroStateManager
        Petrophysical state tracker
    mvars : ModelStateManager
        Model state with geometry and signed distances
    metrics : InversionMetrics
        Metrics tracker
    gpars : GeometryParameters
        Grid geometry parameters
    spars : SaveParameters
        Save/output parameters
    resid_vect : ndarray
        Current data residual vector
    geophy_data : GeophysicalData
        Observed geophysical data
    sensit : Sensitivity
        Sensitivity matrix for forward modeling
    log_run : Logger
        Logger for operation details
    i : int
        Current iteration number
        
    Returns
    -------
    success : bool
        True if birth operation completed successfully
    birth : bool
        True if a new unit was actually birthed
    resid_vect : ndarray
        Updated residual vector after birth
        
    Notes
    -----
    Birth process:
    1. Compute gradient of data misfit
    2. Identify connected high-gradient regions
    3. Evaluate nucleation candidates
    4. Optimize petrophysical value for best fit
    5. Add to signed distances via petrovals.insert_value()
    6. Recalculate phi_curr with new unit
    """

    # TODO: for birth, it'll be necessary to calculate the signed distances of new model to calculate the prior model cost. 

    # Get the domains for birth. 
    birth_params.calc_births_domains(mvars.m_curr) 

    mvars.m_curr, metrics.data_misfit[i], densconst_birth, resid_vect, birth = \
        _birth_deterministic(birth_params, metrics.data_misfit[i-1], sensit, 
                            geophy_data, resid_vect, mvars, log_run)
    
    if birth and (densconst_birth != None):
        # Insert petro value of new unit into the tracked array. 
        new_id = petrovals.insert_value(densconst_birth)
        # Calculate signed distances of new model for calculation of prior model cost. 
        mvars.phi_curr = tu.calc_signed_distances(mvars.m_curr, petrovals.all_values, cell_size=gpars.cell_sizes, narrow=False)
        log_run.info(f"Birth with value = {densconst_birth} for with index {new_id}")
    else: 
        log_run.info(f"No birth!!")
        metrics.data_misfit[i] = metrics.data_misfit[i-1]
        birth = False
    
    if DEBUG:
        if not qc.birth_pass_sanity(metrics, mvars, i, petrovals, max_misfit=MAX_MISFIT, callback_printer=log_run.info):
            om.save_model_to_vtk(mvars.m_curr.flatten(), gpars, 
                                filename=spars.path_output + '/' + spars.filename_model_save_rt + str(i), 
                                save=spars.save_plots, log_run=log_run)
            return False, birth, resid_vect
    
    return True, birth, resid_vect


def _calc_metrics_and_decide(petrovals, mvars, metrics, shpars, rng_main, geophy_data, 
                             sensit, log_run, i, pert_type, birth_occured, death_occured, kill_too_weak):
    """
    Calculate forward model, metrics, and make accept/reject decision.
    
    Returns:
        bool: accept - whether the proposal should be accepted
    """
    # Calculate forward data and metrics
    fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)
    metrics.calc_data_rms_misfit(geophy_data, ind=i) # TODO Move this to metrics class? 

    # First check cases of automatic rejection. 
    if (pert_type == 3) and (not birth_occured):
        accept = False
        if log_run is not None:
            log_run.info('No birth could occur - Auto-reject model since no change')
            log_run.info(f"Model it. {i}, Pert type {pert_type}: REJECTED. ")
            # metrics.accept_ratio[i] = 0.
        return accept
    
    # TODO: terms of posterior should be calculated each time!

    if (pert_type == 4) and (not death_occured):
        accept = False
        if log_run is not None:
            log_run.info('No death could occur - Auto-reject model since no change')
            log_run.info(f"Model it. {i}, Pert type {pert_type}: REJECTED. ")
            # metrics.accept_ratio[i] = 0.
        return accept
    
    # Calculate terms of log-likelihood and acceptance ratio
    metrics.calc_log_likelihood_ratio(shpars.std_data_fit, ind=i, log_run=log_run)

    if (pert_type == 0) or (pert_type == 1):
        if log_run is not None:
            log_run.info('TODO fix prior model cost')

    # Calculate petrohysical prior ratio
    metrics.calc_log_priorpetro_ratio(petrovals, ind=i, log_run=log_run)
    
    # Calculate geometrical misfit. 
    metrics.calc_log_priorgeom_ratio(petrovals, mvars.phi_curr, mvars.phi_prior, 
                                     std_geom_glob=shpars.std_geom_glob,  
                                     inv_var=shpars.local_weights_prior, 
                                     log_run=log_run, 
                                     ind=i, pert_type=pert_type)
    
    # Calculate the log-posterior. 
    metrics.calc_log_posterior(i, log_run)
    
    # Calculate acceptance ratio and make decision
    accept, force_accept = metrics.accept_proposal(rng_main, 
                                                   shpars.force_pert_dict, 
                                                   i, pert_type,
                                                   override_force=kill_too_weak)
    if force_accept and (log_run is not None):
        log_run.info(f"forced accept for pert_type={pert_type}")

    # Logging.
    # Make accept/reject decision.
    if accept:
        if force_accept or kill_too_weak:
            decision_text = "ACCEPTED, FORCED"
        else: 
            decision_text = "ACCEPTED"
    else:
        decision_text = "REJECTED"

    if log_run is not None:
        log_run.info(f"Model it. {i}, Pert type {pert_type}: {decision_text}. "
                        f"accept_ratio={metrics.accept_ratio[i]:.4f}, "
                        f"data_misfit={metrics.data_misfit[i]:.4f}, "
                        f"model_misfit={metrics.model_misfit[i]:.4f}")
        log_run.info(f"Force accept: {force_accept}")
    
    return accept


def _update_accepted_state(metrics, petrovals, mvars, shpars, pert_type, 
                        birth_occured, n_births, gpars, spars, i, log_run):
    """
    Handle all the logic when a proposal is accepted.
    
    Returns:
        int: updated n_births count
    """
    metrics.last_misfit_accepted = metrics.data_misfit[i]
    metrics.it_accepted_type.append(pert_type)
    metrics.it_accepted_model.append(i)

    if (pert_type == 0) and (shpars.force_pert_type in ('petrophy_increase', 'petrophy_decrease')):
        petrovals.set_prev_as_current()
        mvars.mod_aux[mvars.mod_aux == petrovals.all_values[shpars.ind_unit_force]] = petrovals.all_values[shpars.ind_unit_force]     
        
    if pert_type != 2:
        if shpars.force_pert_type is not None:
            mvars.mod_aux[mvars.mod_aux == petrovals.all_values[shpars.ind_unit_force]] = petrovals.all_values[shpars.ind_unit_force]        
            mvars.phi_aux = tu.calc_signed_distances_opti(mvars.mod_aux, mvars.mod_aux_prev, mvars.phi_aux_prev, 
                                                        np.array([petrovals.all_values[shpars.ind_unit_force]]), 
                                                        cell_size=gpars.cell_sizes, narrow=True, log_run=log_run)
        # mvars.phi_aux[mvars.phi_aux > 0] = + np.inf
    elif pert_type == 2: 
        # TODO fix the fact that rock indices can change during modelling. 
        petrovals.set_prev_as_current()
        if shpars.force_pert_type is not None:
            mvars.mod_aux[mvars.mod_aux == petrovals.all_values[shpars.ind_unit_force]] = petrovals.all_values[shpars.ind_unit_force]        

    if pert_type == 3:
        if birth_occured: 
            n_births += 1
            petrovals.resync_with_model(mvars.m_curr)  # Update petro with birth.
            # mvars.phi_curr = tu.calc_signed_distances_opti(mvars.m_curr, mvars.m_prev, mvars.phi_prev, petrovals.all_values, 
            #                                 cell_size=gpars.cell_sizes, birth=birth, narrow=True, log_run=log_run)
            mvars.phi_curr = tu.calc_signed_distances(mvars.m_curr, petrovals.all_values, cell_size=gpars.cell_sizes, narrow=False)  # TODO restrict only to affected units.

            # Should re-calc mvars.phi_aux only if it affected by the pert? TODO make sure this bit is where it should be???
            if shpars.force_pert_type in ('petrophy_increase', 'petrophy_decrease'):
                mvars.phi_aux = tu.calc_signed_distances_opti(mvars.mod_aux, mvars.mod_aux_prev, mvars.phi_aux_prev, 
                                                              np.array([petrovals.all_values[shpars.ind_unit_force]]), 
                                                              cell_size=gpars.cell_sizes, narrow=True, log_run=log_run)

    if pert_type == 4:  
            n_births -= 1 
            petrovals.resync_with_model(mvars.m_curr)
            mvars.phi_curr = tu.calc_signed_distances(mvars.m_curr, petrovals.all_values, cell_size=gpars.cell_sizes, narrow=False)  # TODO restrict only to affected units.

    # Save accepted model every shpars.save_interval.
    if i % spars.save_interval == 0: 
        om.save_model_to_vtk(mvars.m_curr, gpars, 
                            filename=spars.path_output + '/' + spars.filename_model_save_rt + str(i), 
                            save=spars.save_plots, log_run=log_run)
    # TODO save also the fwd for each model? As attached to the model file?
    
    return n_births


def _restore_previous_state(mvars, petrovals, metrics, pert_type, force_pert_type, index_tmp, 
                        pert_index_force, i, geophy_data, sensit):
    """
    Restore state of previous model for model and petrophysical tracker. 
    """
    # Sanity check.
    mvars.assert_shape_dtype(DEBUG)
    # Discard proposed pertubations and revert to previous model. 
    mvars.revert_to_prev()
    # Re-calculate data misfit. TODO: store previous instead of re-calc? 
    fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)

    if (pert_type == 2) or (pert_type == 3):
        # Revert petrophy values to previous ones.
        petrovals.revert_to_prev()
        if (pert_type == 2):
            if force_pert_type in ('petrophy_increase', 'petrophy_decrease'):
                if (index_tmp == pert_index_force):
                    petrovals.pert = petrovals.get_value_by_orig_index(pert_index_force)['val']

    if (pert_type == 0) and (force_pert_type in ('petrophy_increase', 'petrophy_decrease')):
        petrovals.revert_to_prev()

    # No need to revert the petro for pert_type == 4 because it's not been changed yet. 
    # if (pert_type == 4):

    if i != 0: 
        metrics.log_priorgeom_ratio[i] = metrics.log_priorgeom_ratio[i-1]
        metrics.log_priorpetro_ratio[i] = metrics.log_priorpetro_ratio[i-1]
        # metrics.petro_misfit[i] = metrics.petro_misfit[i-1]
        metrics.log_likelihood_ratio[i] = metrics.log_likelihood_ratio[i-1]
        # Not backtracking the mistifs to keep track of evolution across chain. 


def petro_index_force_pert(force_pert_type, petrovals):
    # TODO move this to transd_utils? 
    if force_pert_type is not None: 
        return int(np.where(petrovals.pert[petrovals.pert != 0.] == petrovals.all_values)[0])
    else: 
        return None


def _apply_death(mvars, petrovals, geophy_data, sensit, log_run=None):
    """
    Remove weak or poorly-fitting geological units from the model.
    
    Strategy:
    - If a birthed unit is critically weak (< min threshold), force kill it
    - Otherwise, compare killing the two smallest birthed units and pick the 
      one that reduces data misfit most
    - Original units are protected and never removed
    
    Parameters
    ----------
    mvars : ModelStateManager
        Model state with current geometry and signed distances
    petrovals : PetroStateManager
        Petrophysical state tracker
    geophy_data : GeophysicalData
        Observed geophysical data
    sensit : Sensitivity
        Sensitivity matrix for forward calculation
    log_run : Logger, optional
        Logger for operation details
        
    Returns
    -------
    death_occurred : bool
        True if a unit was killed, False otherwise
    success : bool
        True if operation completed without errors
        
    Notes
    -----
    Updates mvars.m_curr, mvars.phi_curr, and petrovals tracking in-place.
    Only birthed units can be killed; original units are always protected.
    """

    def _kill_unit(src_model, dst_model, val_to_kill_):
        """
        Replace unit `val_to_kill` in `src_model` with its largest contact.
        Writes result in-place to `dst_model`.

        Usage:  -- src_model: the current model. 
                -- dst_model: a temporary model used to generate candidate model.
                -- val_to_kill_: density value to remove from the model.
        """
        # Copy source to destination
        np.copyto(dst_model, src_model)
        # Determine which value to replace with
        petro_largest_contact_, _ = tu.calculate_contacts_for_unit(src_model, val_to_kill_, mode="largest")
        # In-place replacement
        dst_model[src_model == val_to_kill_] = petro_largest_contact_

    def _evaluate_sosq_misfit(model, sensit, geophy_data):
        """Compute sum of square misfit for a given model."""
        resp = fcu.calc_fwd(sensit, model, unit_conv=geophy_data.unit_conv)
        return np.sum((geophy_data.data_field - resp) ** 2)

    success = False

    # Check whether to kill the smallest unit - except if it's from the original units.
    kill_too_weak, ind, val_to_kill, item_id = petrovals.check_units_health(model_curr=mvars.m_curr, 
                                                                min_cells_birthed=150,
                                                                protect_original=True)
    
    if kill_too_weak:

        # Get contact information for unit to kill. TODO check this againts tu.calculate_unit_contacts_fast
        petro_largest_contact, _ = tu.calculate_contacts_for_unit(mvars.m_curr, val_to_kill, mode="largest")

        # Replace unit to kill by unit with largest contact. 
        mvars.m_curr[mvars.m_curr == val_to_kill] = petro_largest_contact

        if log_run is not None: 
            log_run.info(f"\n---- death_candidates not empty death_candidates: ind = {ind}, val_to_kill = {val_to_kill}, item_id = {item_id}")
            log_run.info(f"Propose model with killed unit {ind} -- it is the weakest!!")

        success = True
        
    else: 
        
        # Identify the (up to) two smallest birthed units. 
        smallest = tu.get_smallest_birthed_units(petrovals, mvars, n=2)

        # Propose two models, killing, respectively, the two smallest birthed units. 
        if len(smallest) == 2:
            # Get information bout units to kill. 
            ind1, val_to_kill_1, _, id1 = smallest[0]
            ind2, val_to_kill_2, _, id2 = smallest[1]

            # Kill the smallest unit. 
            _kill_unit(mvars.m_curr, mvars.m_tmp_1, val_to_kill_1)
            # Calculate sum of square misfit. 
            rms_tmp_1 = _evaluate_sosq_misfit(mvars.m_tmp_1, sensit, geophy_data)

            # Kill the second smallest unit. 
            _kill_unit(mvars.m_curr, mvars.m_tmp_2, val_to_kill_2)
            # Calculate sum of square misfit. 
            rms_tmp_2 = _evaluate_sosq_misfit(mvars.m_tmp_2, sensit, geophy_data)

            # Keep as current the model with better misfit. 
            np.copyto(mvars.m_curr, mvars.m_tmp_1 if rms_tmp_2 > rms_tmp_1 else mvars.m_tmp_2)

            if log_run is not None:
                log_run.info(f"Propose normal death: Choose between 2 scenarios. ")
                log_run.info(f"---- death_candidate 1: ind = {ind1}, val_to_kill = {val_to_kill_1}, item_id = {id1}")
                log_run.info(f"---- death_candidate 2: ind = {ind2}, val_to_kill = {val_to_kill_2}, item_id = {id2}")
            
        # If only one birthed unit -- proposed killing it. 
        elif len(smallest) == 1:
            if log_run is not None:
                log_run.info(f"Propose normal death: Single Scenario. ")
            ind, val_to_kill, _, item_id = smallest[0]
            _kill_unit(mvars.m_curr, mvars.m_tmp, val_to_kill)
            np.copyto(mvars.m_curr, mvars.m_tmp)
        
        success = True

    return kill_too_weak, success, True


def pertub_scalar_fields(petrovals, mvars, metrics, shpars, gpars, phipert_config, birth_params, run_config, spars, rng_main, 
                          geophy_data, sensit, log_run):
    if DEBUG: 
        log_run.info("Using DEBUG: execution a bit slower!")

    petrovals.pert = petrovals.pert[1]
    force_pert_type = shpars.force_pert_type

    # Get index of units corresponding to pert.
    pert_index_force = petro_index_force_pert(force_pert_type, petrovals)
    
    n_births = 0  # No birth occured at the beginning of sampling.
    n_births_max = birth_params.n_births_max  

    accept = True

    for i in range(shpars.num_epochs):  # Would starting wth i==1 make more sense? 
        log_run.info(f"\n---- STARTING ITERATION {i} ----")
        birth_occured = False 
        death_occured = False
        kill_too_weak = False
        index_tmp = None
        # death = False 
        # force_accept = False  
        # break_exec = False

        # Init resid vect. 
        resid_vect = fcu.calc_resid_vect(geophy_data.data_field, geophy_data.data_calc)

        # Update previous model if accepted: copy current model as previous one. 
        mvars.update_prev(accept)
        petrovals.update_prev(accept) 

        if DEBUG:
            if not qc.validate_model_state_consistency(mvars, petrovals, raise_error=True, 
                    callback_printer=log_run.error):
                log_run.error("Validation failed - marking operation as unsuccessful")
                success = False

        pert_type = st.get_perturbation_type(rng_main, run_config, n_births, n_births_max, log_run)  # TODO move this with the shpars class?

        # GUIDED/FORCED PERTURBATION. Ex.: growing the anomaly inside unit.
        if (pert_type == 0) and (force_pert_type is not None):
            success = _apply_forced_perturbation(petrovals, mvars, metrics, shpars, spars, gpars, phipert_config, rng_main, 
                                                geophy_data, sensit, log_run, i)
            
        # RANDOM GEOMETRICAL PERTURBATION. Homogenous 'Normal' pertubation of unit as a whole with random field everywhere. 
        elif (pert_type == 1): 
            success = _apply_geometrical_perturbation(petrovals, mvars, metrics, shpars, spars, phipert_config, gpars, rng_main, 
                                                    geophy_data, sensit, log_run, i)
            
        # PETROPHYSICAL PERTURBATION. 
        elif (pert_type == 2): 
            success, index_tmp = _apply_petrophysical_perturbation(
                petrovals, mvars, metrics, shpars, pert_index_force, rng_main, 
                geophy_data, sensit, log_run, i)

        # BIRTH OF A UNIT. TODO make it conditional on a value of the score (derivative of likelihood?)
        elif (pert_type == 3): 
            success, birth_occured, resid_vect = _apply_birth(birth_params,
                petrovals, mvars, metrics, gpars, spars, resid_vect, geophy_data, sensit, log_run, i)

        # DEATH OF A UNIT. 
        elif (pert_type == 4):
            if petrovals.has_birthed_units():
                kill_too_weak, success, death_occured = _apply_death(mvars, petrovals, geophy_data, sensit, log_run)
            else: 
                death_occured = False
                success = True
                kill_too_weak = False
                if log_run is not None:
                    log_run.info("No birthed units available for death.\n"
                                f"  Current units: {petrovals.n_units_total} (all original)\n")
                
        if not success: 
            om.save_model_to_vtk(mvars.m_curr, gpars, 
                    filename=spars.path_output + '/' + spars.filename_model_save_rt + str(i), 
                    save=spars.save_plots, log_run=log_run)
            qc.raise_perturbation_error(pert_type, i, log_run)
            return None

        # Get accept ratio.
        accept = _calc_metrics_and_decide(petrovals, mvars, metrics, shpars, rng_main, geophy_data, 
                                                      sensit, log_run, i, pert_type, birth_occured, death_occured, kill_too_weak)

        if accept: 
            n_births = _update_accepted_state(metrics, petrovals, mvars, shpars, pert_type, 
                                    birth_occured, n_births, gpars, spars, i, log_run)
        elif not accept: 
            _restore_previous_state(mvars, petrovals, metrics, pert_type, force_pert_type, index_tmp, 
                                    pert_index_force, i, geophy_data, sensit)
            
        metrics.n_units_total[i] = petrovals.n_units_total
        
        # if i % 10 == 0:
        #     metrics.track_contacts(mvars.m_curr, gpars.cell_sizes, i)
       
    return None