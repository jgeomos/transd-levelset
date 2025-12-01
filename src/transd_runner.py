"""
Module for running  trans-d inversion.

Main workflow orchestrator for trans-dimensional inversion.
Coordinates data loading, state initialization, and sampling.

This module initializes model parameters, loads sensitivity kernels and data,
sets up solver configurations, and executes the main inversion loop. It acts as
the high-level controller that coordinates model preparation, perturbation,
and forward calculations using the Tomofast-X framework.

Main entry point:
    run_nullspace_navigation(par, log_run, rng_main, run_config, sensit=None, folder_transd="")

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

# Standard libs. 
import numpy as np
import os
from importlib import reload

# Home-made libs. 
import src.forward_solver.forward_calculation as fcu
from src.forward_solver.tomofast_sensit import TomofastxSensit

import src.input_output.tomofast_reading_utils as tu
import src.input_output.input_params as input_params
import src.input_output.output_manager as om

import src.utils.plot_utils as ptu

import src.transd_solver as ts
import src.petro_state as psm
import src.model_state as ms
import src.inversion_metrics as im
import src.birth_death as bd

# matplotlib.use('Qt5Agg')

# Possible improvements:
# 1. find a more efficient way to unpack values from the <par> argument in the solve function.
# 2. add the capability to also load a sensitivity matrix calculated with UBC codes or Simpeg.
# 3. give more flexibility to the mask (i.e.: add the possibility to load a mask externally).

# Reload modules to avoid restarting kernel each time a modif is made.
reload(ts)
reload(input_params)
reload(ptu)
reload(ms)
reload(fcu)
reload(tu)
reload(im)
reload(psm)
reload(bd)


def run_transd(par, log_run, rng_main, run_config, sensit=None, folder_transd = ""):
    """
    Execute the trans-dimensional inversion workflow.

    Orchestrates the inversion process: loads data and sensitivity matrices,
    initializes model and petrophysical state, configures MCMC parameters and runs
    sampling.

    Note: this function has many output arguments so they can be used later on from the workspace. 
    
    Parameters
    ----------
    config : InputParameters
        Configuration with file paths and solver parameters
    log_run : Logger
        Logger instance for recording execution
    rng_main : numpy.random.Generator
        Random number generator for reproducibility
    run_params : RunBaseParams
        Expert-level algorithm configuration
    sensit : Sensitivity, optional
        Pre-loaded sensitivity matrix (if None, loaded from disk)
    data_dir : str, optional
        Root directory for input data files
        
    Returns
    -------
    tuple
        (petrovals, birth_params, mvars, metrics, shpars, gpars, 
         phipert_config, geophy_data, sensit, spars, config)
        Complete set of state managers and configuration objects
        
    Notes
    -----
    This function:
    1. Loads models, gravity data, and sensitivity matrices
    2. Initializes state managers (ModelStateManager, PetroStateManager)
    3. Sets up MCMC parameters and birth/death configuration
    4. Executes main inversion loop (pertub_scalar_fields)
    5. Returns all components for post-processing
    
    The inversion uses level set methods for representing geological units
    and implements trans-dimensional sampling (variable number of units).
    """

    cwd = os.getcwd() 
    folder_transd = os.path.join(cwd, folder_transd)

    bouguer_anomaly = True

    #----- Unpacking parameter class 'par'.
    # General parameters. 
    sensit_path = par.sensit_path
    tomofast_sensit_nbproc = par.get_tomofast_sensit_nbproc(path=folder_transd)
    unit_conv = par.unit_conv
    use_mask_domain = par.use_mask_domain

    # Parameters for Hamiltonian nullspace navigation (continuous values). 
    std_geom_glob = par.std_geom_glob  # TODO Make use of this param.
    sensit_type = par.sensit_type

    data_vals_filename = par.data_vals_filename
    data_background_filename = par.data_background_filename
    perturbation_filename = par.perturbation_filename
    mask_filename = par.mask_filename
    model_filename = par.model_filename

    # Init. parameters for descrete value nullspace navigation (geometry case). 
    # Parameters for the generation of random fields using pink noise. 
    # Correlation lengths along z, x, y dir. TODO: check that it's z, x, y direction. 
    correlation_length_z = par.correlation_length_0
    correlation_length_x = par.correlation_length_1
    correlation_length_y = par.correlation_length_2
    correlation_length = (correlation_length_z, correlation_length_x, correlation_length_y)
    rotation_angle = np.array((par.rotation_angle_0, par.rotation_angle_1, par.rotation_angle_2))  # Degrees. 

    n_births_max = par.n_births_max

    log_run.info(f"correlation_length: {correlation_length}")

    # Sanity check on some input parameters. 
    if unit_conv==True and sensit_type=="magn": 
        log_run.info("Prompting user to acknowledge unit_conv will be set to False.")
        input("unit_conv will be set to False.\n\nPress Enter to continue...\n")
        log_run.info("User continued after unit conversion warning.")

    # Setup for saving plots and saving of files and models for posterior analysis.
    spars = input_params.SavePars(path_output= par.path_output, 
                                    filename_model_save_rt='m_curr', 
                                    filename_aux_save_rt='mod_aux',
                                    save_plots=par.save_plots,
                                    save_interval = par.save_interval)  # TODO Should SavePars go in output_manager instead of input_params?
    
    # Get normalized cell size ratios.
    # [spacing_z, spacing_y, spacing_x] = gpars.get_spacing(normalise=True)
    spacing_z, spacing_y, spacing_x = 1., 1., 1.  # TODO add this to the parfile??
    cell_sizes = [spacing_z, spacing_y, spacing_x] 
    log_run.info(f"cell_sizes: {cell_sizes}")

    # Initialize model parameters class and set the dimensions of the model (nz, nx, ny).
    gpars = input_params.GridParameters(spacing_x=spacing_x, 
                                        spacing_z=spacing_z, 
                                        spacing_y=spacing_y)
    gpars.get_tomofast_model_gridsize(model_filename, path=folder_transd)
    # Initialise model variables class.
    mvars = ms.ModelStateManager(dim=gpars.dim)

    # ----------------------------------------------------------------------------------
    # Read models and model grid.
    local_weights_filename = par.local_weights_filename
    # mvars.m_start: starting point for the nullspace navigation (unperturbed model).
    mvars.m_start, gpars = tu.read_tomofast_model(model_filename, gpars, path=folder_transd)

    local_weights_prior, _ = tu.read_tomofast_model(local_weights_filename, gpars, path=folder_transd)  

    # JG 03/09
    # mvars.m_start[mvars.m_start == 2660.] = 2685.

    # ----------------------------------------------------------------------------------
    # Pre-processing parameters: definition of mask.
    # Index of rock unit for perturbation and null space analysis (rocks indexed by increasing density).
    ind_unit_mask = par.ind_unit_mask  # Index used for the calculation of the mask. In Pyrenees paper: 9 = Mantle.
    # Distance max in number of cells away from the outline of rock unit with index 'ind_unit_mask'.
    distance_max = par.distance_max  # In Pyrenees paper: 8 in tests shown.
    # ----------------------------------------------------------------------------------
    # Calculate the domain mask: cells where the solver is allowed to change the model values.
    # The domain mask is used to reduce the part of the model that can be affected by null space navigation.
    # We define the compute domain with masks on distfance to the outline of selected unit (if applicable).
    mvars.set_masked_domain(use_mask_domain, distance_max, ind_unit_mask, mvars.m_start, mask_first_layers=False) 
    # TODO: read mask_first_layers from the input files.
    # ----------------------------------------------------------------------------------
    #Reading the geophy data.
    # Initialise geophy data class.
    geophy_data = fcu.GeophyData(unit_conv=unit_conv, bouguer_anomaly=bouguer_anomaly)

    tu.read_tomofast_data(geophy_data, data_vals_filename, data_type='field', path=folder_transd)
    tu.read_tomofast_data(geophy_data, data_background_filename, data_type='background', path=folder_transd)

    # ----------------------------------------------------------------------------------
    # Load sensitivity kernel from Tomofast-x.
    if sensit is None:
        sensit = tu.load_sensit_from_tomofastx(sensit_path, 
                                            nbproc=tomofast_sensit_nbproc, 
                                            empty_sensit_class=TomofastxSensit,
                                            type=sensit_type, 
                                            verbose=False, 
                                            path=folder_transd)
        sensit.precompute_all()  # Calculate inverse weights for fwd data calc and cache the transpose of sensit.

    # ----------------------------------------------------------------------------------
    # Initialise solver parameters parameters. TODO: adjust for new version.
    shpars = input_params.SolverParameters( indices_unit_pert=par.indices_unit_pert,
                                            ind_unit_force=par.ind_unit_force,
                                            ind_unit_ref=par.ind_unit_ref,
                                            force_pert_type=par.force_pert_type,
                                            use_dynamic_mask=par.use_dynamic_mask,
                                            std_data_fit=par.std_data_fit,
                                            num_epochs=par.num_epochs,
                                            local_weights_prior=local_weights_prior,
                                            std_geom_glob=std_geom_glob,
                                            use_loaded_mask=par.use_loaded_mask, 
                                            force_pert_0=par.force_pert_0,
                                            force_pert_1=par.force_pert_1,
                                            force_pert_2=par.force_pert_2,
                                            force_pert_3=par.force_pert_3,
                                            force_pert_4=par.force_pert_4,
                                            )  

    # Load model perturbation: the model change we want to impose (the final model should have this change).
    mvars.delta_m_orig, _ = tu.read_tomofast_model(perturbation_filename, gpars, path=folder_transd)

    # Load the mask. 
    if par.use_loaded_mask: 
        mvars.loaded_mask, _ = tu.read_tomofast_model(mask_filename, gpars, path=folder_transd)

    # Initialise models for sampling.
    mvars.m_curr = mvars.m_start.copy()
    mvars.init_tmp()

    # Calculate the geophy data.
    fcu.calc_geophy_data(geophy_data, sensit, mvars.m_curr)
    _, starting_misfit = fcu.calc_data_rms(geophy_data)
    if log_run is not None: 
        log_run.info(f'Data misfit (before perturbation) = {starting_misfit}')
    
    #-----------------------------------------------------------------------------------
    # Initialisation. 
    #-----------------------------------------------------------------------------------
    # Initialise metrics for monitoring. 
    metrics = im.InversionMetrics(reference_misfit=0.,  # TODO read this from the parfile (useful for nullspace)
                                starting_misfit=starting_misfit,                                   
                                n_proposals=shpars.num_epochs, 
                                max_misfit_force=5.0)

    # Initialise petrophysical variables class.
    petrovals = psm.PetroStateManager(std_petro=par.std_petro, 
                                    distro_type='gaussian',
                                    model_curr=mvars.m_curr, 
                                    model_with_pert=mvars.delta_m_orig, 
                                    log_run=log_run)

    # Initialise signed distances. 
    mvars.init_signed_distances(petrovals, cell_size=gpars.cell_sizes)
    mvars.init_prev_model()

    # om.save_model_tomofast(model_filename, np.ones_like(mvars.m_start), "data/models/local_weights_prior.txt")
    shpars.init_weights_prior(local_weights_prior, normalise=False)

    dimensions=np.shape(mvars.phi_curr)

    # Init of a class for the perturbation of signed distances. 
    phipert_config = input_params.PertSignedDistPars(factor_spectrum_min=par.factor_spectrum_min, 
                                                    factor_spectrum_max=par.factor_spectrum_max, 
                                                    amplitude_pert_min=par.amplitude_pert_min,
                                                    amplitude_pert_max=par.amplitude_pert_max,
                                                    dimensions=dimensions, 
                                                    mask=mvars.domain_mask,                                 
                                                    weights=par.weights,
                                                    normalise=par.normalise, 
                                                    correlation_length=correlation_length, 
                                                    corr_zx=par.corr_zx, 
                                                    corr_zy=par.corr_zy, 
                                                    corr_xy=par.corr_xy,
                                                    rotation_angle=rotation_angle, 
                                                    cell_sizes=cell_sizes, 
                                                    rng=rng_main)

    # Init of param class for births. 
    birth_params = bd.BirthConfig(n_blobs_max=4, 
                               n_line_search=10, 
                               min_cells_num=100, 
                               flag_conform_geol=2, 
                               model_dim=mvars.dim,                                    
                               bounds=(-100, 100),
                               rng_seed=None, 
                               nucl_geol=2, 
                               n_births_max=n_births_max, 
                               type_birth=0, 
                               exclude_indices=None)
    
    # Save models prior to sampling.
    om.save_model_to_vtk(mvars.m_curr, gpars, filename=spars.path_output + '/' + 'orig_m_curr', save=spars.save_plots)
    om.save_model_to_vtk(mvars.delta_m_orig, gpars, filename=spars.path_output + '/' + 'delta_m_orig', save=spars.save_plots)
    om.save_model_to_vtk(mvars.mod_aux, gpars, filename=spars.path_output + '/' + 'mod_aux', save=spars.save_plots)


    # Perturbations for the scalar fields. 
    ts.pertub_scalar_fields(petrovals, mvars, metrics, shpars, gpars, phipert_config, birth_params, run_config, spars, rng_main, geophy_data, sensit, log_run)

    # Save last model using Tomofast-x format. 
    # om.save_model_tomofast(model_filename, mvars.m_curr, destination_file="data/models/result_last_model.txt")

    return petrovals, birth_params, mvars, metrics, shpars, gpars, phipert_config, geophy_data, sensit, spars, par

sensit = None
par = None