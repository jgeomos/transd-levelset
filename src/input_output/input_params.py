"""
Input parameter management for trans-dimensional geophysical inversion.

This module handles reading, validating, and storing all configuration parameters
required for running the inversion. Parameters are read from text-based parameter
files (parfiles) that specify model setup, data, sampling options, and output paths.

Main Components
---------------
read_input_parameters : function
    Parse parfile and return InputParameters object
InputParameters : class
    Container for all inversion configuration
RunBaseParams : class
    Runtime configuration (output paths, verbosity, etc.)

Notes
-----
- All file paths are resolved relative to the parfile location
- Missing required parameters will raise ParameterValidationError
- Default values are provided for optional parameters

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"


import configparser
from dataclasses import dataclass, field
import numpy as np
import os
import re
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.simplefilter("always", category=UserWarning)  # Make sure warnings are actually printed when using Jupyter Notebook.


@dataclass(frozen=True)
class RunBaseParams():
    """
    Class containing expert-level parameters.
    These values are considered constants and should not be changed unless modifying core logic.
    """
    # =====
    # TODO move these to parfile!!!
    # =====
    # Debug and system constants
    # debug_mode: bool = True
    # max_misfit_threshold: float = 50.0

    logging: bool = True

    # What index of perturbation is a birth. 
    pert_is_birth: int = 3

    force_pert_type: str | None = None

    # The different types of perturbations. 
    # perts_all = (1, 5)

    def __post_init__(self):
        value = (0, 5) if self.force_pert_type is not None else (1, 5)
        object.__setattr__(self, 'perts_all', value)
    
    # Message mappings.
    # TODO for flexibility: instaed of sampling from 0, 1, 2, ... and mapping to a type, use 
    # dictionnaries with "birth", "death", "petro", "geome" --> move flexibility, less bugs. 
    perturbation_messages: dict = field(default_factory=lambda: {
        0: "pert_type = 0 -> Forcing pre-defined perturbation",
        1: "pert_type = 1 -> Random geometrical perturbation", 
        2: "pert_type = 2 -> Random petrophysical perturbation",
        3: "pert_type = 3 -> Birth of a unit",
        4: "pert_type = 4 -> Death of a unit",
    })


@dataclass(slots=True)
class SavePars:
    """
    Class containing information controlling how or if results will be saved.
    """

    save_plots: bool = False
    path_output: str | Path | None = None
    save_interval: int = None

    filename_model_save_rt: str = "m_curr"
    filename_aux_save_rt: str = "mod_aux"

    def __post_init__(self):
        """Create output folder only if path_output is provided and non-empty."""
        if self.path_output and str(self.path_output).strip():
            self.create_output_folder()
        else:
            raise Exception("⚠️ No output path provided — results will not be saved to disk.")

    def create_output_folder(self):
        """Create the output folder if it doesn’t already exist."""
        folder_path = Path(self.path_output)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output folder created at: {folder_path.resolve()}")  # TODO add this to log_run??


@dataclass(slots=True)
class PertSignedDistPars:

    factor_spectrum_min: float
    factor_spectrum_max: float
    amplitude_pert_min: float
    amplitude_pert_max: float
    dimensions: Tuple[int, ...]
    mask: np.ndarray
    normalise: bool
    correlation_length: Tuple[float, float, float]
    corr_zx: float
    corr_zy: float
    corr_xy: float
    rotation_angle: Tuple[float, float, float]
    cell_sizes: np.ndarray
    rng: np.random.Generator
    weights: np.ndarray | None = None
    fft_class: object | None = None
    rotation_matrix: np.ndarray | None = None

    def __post_init__(self):
        # Validation
        if self.factor_spectrum_min <= 0:
            raise ValueError("factor_spectrum_min must be positive")
        if self.factor_spectrum_min > self.factor_spectrum_max:
            raise ValueError("factor_spectrum_min cannot be greater than factor_spectrum_max")
        if self.weights is None:
            # Default weights to ones (matching spatial dimensions)
            self.weights = np.ones(self.dimensions[1:])

        # Optional: validate mask and dimensions match
        if self.mask.shape != self.dimensions[1:]:
            warnings.warn(f"self.mask.shape = {self.mask.shape}", UserWarning)
            warnings.warn(f"self.dimensions[1:] = {self.dimensions[1:]}", UserWarning)
            warnings.warn("mask shape must match spatial dimensions", UserWarning)


@dataclass(slots=True)
class GridParameters:
    """
    A class that initialises the parameters of the 3D mesh used for the voxet.
    """

    # Coordinates of centres of model cells.
    x: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    y: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    z: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    # Coordinates of the edges of model cells.
    x1: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    x2: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    y1: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    y2: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    z1: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    z2: np.ndarray = field(default_factory=lambda: np.array([1.0]))
   
    # Number of cells in x, y, and z direction.
    dim: tuple[int, int, int] = (-1, -1, -1) # or make it tuple since dim doesnt change?

    # The spacing between cells along x, y, and z directions, excluding padding. 
    # diff_x: float = 1.0
    # diff_y: float = 1.0
    # diff_z: float = 1.0
    spacing_x: float = 1.
    spacing_y: float = 1.
    spacing_z: float = 1.

    cell_sizes: np.ndarray = field(default_factory=lambda: np.array([1., 1., 1.]))

    def __post_init__(self):
        # Automatically update cell_sizes from spacings
        self.cell_sizes = np.array([self.spacing_x, self.spacing_y, self.spacing_z], dtype=float)

    # Total number of elements calculated from dim.
    # @property
    # def n_el(self):
    #     return np.prod(self.dim)

    def get_tomofast_model_gridsize(self, filename, path=None):
        """
        Read the grid size of the model, i.e. nx, ny, nz, from a Tomofast-x model file. 
        Extract it from the last row of the file. 
        """

        if path is not None: 
            filename = os.path.join(path, filename)

        # Check if the file exists.
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")
        
        model = np.loadtxt(filename, skiprows=1)

        # Get the last line, which contains the dimension of the model.
        last_row = model[-1]

        # Get nx, ny, nz.
        grid_size = last_row[-3:]

        # Convert to integers.
        self.dim = [int(x) for x in grid_size]

        if any(d <= 0 for d in self.dim):
            ValueError("⚠️ Incorrect grid dimensions: negative value found!")

        # Revertse the dimensions order for conversion from Fortran to Python implementation of inversion code.
        self.dim = tuple(self.dim[::-1])
    
    # Get the spacing between cells along x, y, and z directions, excluding padding. 
    def get_spacing(self, normalise: bool = True):
        """Compute and optionally normalise spacing between cells along each axis."""
        if len(self.x) < 2 or len(self.y) < 2 or len(self.z) < 2:
            raise AttributeError("The grid is not fully defined (x, y, z arrays too short).")


    #     # x-dir.
    #     self.diff_x = np.diff(self.x)
    #     self.diff_x = self.diff_x[self.diff_x != 0]
    #     # Get the middle index. 
    #     middle_index = len(self.diff_x.flatten()) // 2
    #     self.diff_x = self.diff_x.flatten()[middle_index]
    #     # self.diff_x = mode(self.diff_x.flatten())[0][0]

    #     # y-dir.
    #     self.diff_y = np.diff(self.y)
    #     self.diff_y = self.diff_y[self.diff_y != 0]
    #     # self.diff_y = mode(self.diff_y.flatten())[0][0]
    #     middle_index = len(self.diff_y.flatten()) // 2
    #     self.diff_y = self.diff_y.flatten()[middle_index]
        
    #     # z-dir.
    #     self.diff_z = np.diff(self.z)
    #     self.diff_z = self.diff_z[self.diff_z != 0]
    #     # self.diff_z = mode(self.diff_z.flatten())[0][0]
    #     middle_index = len(self.diff_z.flatten()) // 2
    #     self.diff_z = self.diff_z.flatten()[middle_index]
        
    #     if normalise: # Such that the spacing is normalised.
    #         # max_diff = np.max([self.diff_x, self.diff_y, self.diff_z])
    #         # self.diff_x /= max_diff
    #         # self.diff_y /= max_diff
    #         # self.diff_z /= max_diff
        return list((self.diff_x, self.diff_y, self.diff_z))


@dataclass(slots=True)
class SolverParameters:
    """
    Class to store hyperparameters for sampling.
    Default values are only indicative and are by no means acceptable for all problems.
    """

    # Index of the unit that will be perturbed using sampling. 
    indices_unit_pert: int = 0
    # Index of unit of which growth will be forced. 
    ind_unit_force: int | None = None
    # Index of unit whose geometry will guide the changes.
    ind_unit_ref: int | None = None
    # Type of perturbation that can be forced: 'petrophy' or 'geometry'.
    force_pert_type: str | None = None
    # Boolean controlling whether dynamic masking will be used. 
    use_dynamic_mask: bool = False
    # Variance on noise for calculation of the likelihood. 
    std_data_fit: float = 0.5
    # Number of time steps.
    num_epochs: int = 100
    # Local weights on prior model.
    local_weights_prior: np.ndarray | None = None
    # Global weights on prior. 
    std_geom_glob: float = 1.
    # Flag for use of mask on domain. 
    use_loaded_mask: bool = False

    # Percentage of forced, used defined changes.
    force_pert_0: float = 0.0
    # Force accept of perth type 1: random geometrical perturbation
    force_pert_1: float = 0.0
    # Force accept of perth type 2: random petrophysical perturbation
    force_pert_2: float = 0.0
    # Force accept of perth type 3: birth of a unit
    force_pert_3: float = 0.0
    # Force accept of perth type 4: death of a unit
    force_pert_4: float = 0.0
    # force_pert dictionnary
    force_pert_dict: dict = None

    def __post_init__(self):
        self.force_pert_dict = {
                                0: self.force_pert_0,
                                1: self.force_pert_1,
                                2: self.force_pert_2,
                                3: self.force_pert_3,
                                4: self.force_pert_4,
                                }
        # Validation: all values must be >= 0.
        for k, v in self.force_pert_dict.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"force_pert[{k}] must be between 0 and 1, got {v}.")

    def init_weights_prior(self, loaded_weights, normalise: bool = True):
        """Initialise and normalise local weights for the prior model."""
        
        if loaded_weights is not None and loaded_weights.size != 0:
            loaded_weights = np.asarray(loaded_weights, dtype=float)
            loaded_weights = loaded_weights.flatten()
        
        if loaded_weights is None:
            loaded_weights = 1. 
            normalise = False 
            warnings.warn("No value for loaded_weights: set it to a scalar = 1. ", UserWarning)

        if normalise:
            wmin, wmax = np.min(loaded_weights), np.max(loaded_weights)
            if np.isclose(wmax, wmin):
                raise ValueError("Loaded weights have zero variance — cannot normalise.")
            self.local_weights_prior = (loaded_weights - wmin) / (wmax - wmin)
            self.local_weights_prior += 0.001  # numerical stability

        self.local_weights_prior = loaded_weights

        # Sanity check
        if np.isnan(self.local_weights_prior).any():
            print(self.local_weights_prior)
            raise ValueError("local_weights_prior contains NaN values!")

        return None
    


@dataclass(slots=True)
class InputParameters:
    """
    A class to contain all input parameters from the Parfile. 
    """
    # -------------------------------
    # Section 'FilePaths'.
    # -------------------------------
    model_filename: str = ""
    perturbation_filename: str = ""
    local_weights_filename: str = ""
    mask_filename: str = ""
    # Geophysical data, e.g., Bouguer anomaly.
    data_vals_filename: str = 'data/gravity_data/data_vals.txt'
    # Value of the background model used, e.g., in the calculation of the Bouguer anomaly.
    data_background_filename: str | None = None
    path_output: str = "output/"
    sensit_path: str = "input/SENSIT/"

    # -------------------------------
    # Section 'SolverParameters'.
    # -------------------------------
    # String of characters determining the type of inversion / sensitivity matrix ('grav' or 'magn').
    sensit_type: str = ""
    # Number of procs used to calculate the sensitivity kernel with Tomofast.
    tomofast_sensit_nbproc: int = 1
    # Flag on unit conversion
    unit_conv: bool = True
    # Flag on whether we use a mask to reduce the domain where modifications of the model are allowed.
    use_mask_domain: bool = True
    # - 2.5e-12 for plunging crust
    # 1.e-11 for mantle chunk in axial zone.
    # Flag for use of mask from file. 
    use_loaded_mask: bool = False

    # Number of time steps.
    num_epochs: int = 350

    # ------------------------------------
    # Section 'PreProcessingParameters'.
    # ------------------------------------
    # Index of rock unit (by increasing density value) to define the mask on perturbations. 9 = Mantle.
    ind_unit_mask: int = 9
    # Distance max in number of cells away from the outline of rock unit considered.
    distance_max: int = 4  # 4, 8 in tests shown in Pyrenees paper.

    # ------------------------------------
    # Section 'SaveOutput'.
    # ------------------------------------
    save_plots: bool = False
    # Interval along the chain ie num of iterations the models will be saved. 
    save_interval: int = 100 
    # ------------------------------------
    # Section 'SamplingParams'.
    # ------------------------------------
    # Index of unit that will be perturbed. 
    indices_unit_pert: List[int] = field(default_factory=list)
    # Index of units whose modifications will be forced. 
    ind_unit_force: int | None = None
    # Index of unit whose geometry will guide the changes.
    ind_unit_ref: int = 0
    # Type of perturbation that can be forced: 'petrophy' or 'geometry'.
    force_pert_type: str | None = None 
    # Whether using dynamic masking. 
    use_dynamic_mask: bool = False
    # Standard deviation for the likelihood. 
    std_data_fit: float = 1.
    # Uncertainty on petrophysical values. 
    std_petro: float = 1.
    # Max number of births.
    n_births_max: int = 5
    # Weight of prior model term.
    std_geom_glob: float = 1.e-1
    # Percentage of forced, user-defined non-reversible changes.
    force_pert_0: float = 0.0
    # Force accept of perth type 1: random geometrical perturbation
    force_pert_1: float = 0.0
    # Force accept of perth type 2: random petrophysical perturbation
    force_pert_2: float = 0.0
    # Force accept of perth type 3: birth of a unit
    force_pert_3: float = 0.0
    # Force accept of perth type 4: death of a unit
    force_pert_4: float = 0.0

    # ------------------------------------
    # Section 'NoiseParams'.
    # ------------------------------------
    # Factor mutliplyplying power of spectrum
    factor_spectrum_min: float = 2.
    factor_spectrum_max: float = 5.
    # Amplitude of noise. 
    amplitude_pert_min: float = 0.75 
    amplitude_pert_max: float = 2.
    # Local weights multiplying the values of noise. Can be used as mask.
    weights: str = None
    # Wether normalise the noise values to be between 0 and 1.  
    normalise: bool = True
    # Correlation lengths along z, x, y dir.
    correlation_length_0: float = 1.
    correlation_length_1: float = 1.
    correlation_length_2: float = 1.
    # Correlation between dimensions. # Between 0. and 1.
    corr_zx: float = 0.0
    corr_zy: float = 0.0
    corr_xy: float = 0.0  
    # Rotation around axes (axis 1, axis 2, axis 3) # Degrees. 
    rotation_angle_0 : float = 0.
    rotation_angle_1 : float = 0.
    rotation_angle_2 : float = 0.

    def get_tomofast_sensit_nbproc(self, path=None):

        first_numbers = []

        if path is None: 
            path_sensit = self.sensit_path
        else:
            path_sensit = os.path.join(path, self.sensit_path)

        for filename in os.listdir(path_sensit):
            match = re.search(r'\d+', filename)
            if match:
                first_numbers.append(int(match.group()))

        unique_val = np.array(np.unique(first_numbers))

        if len(unique_val)>1:
            raise ValueError("Problem in determining the number of CPUs for sensitivity calculation. Check the names or format of sensitivity matrix!")
        
        self.tomofast_sensit_nbproc = unique_val[0]
        return self.tomofast_sensit_nbproc


def parse_value(val):
    val = val.strip()
    return None if val.lower() in ("none", "") else val


def maybeint(value: str):
    value = value.strip()
    if value.lower() == "none":
        return None
    return int(value)


def maybestr(value: str):
    value = value.strip()
    if value.lower() == "none":
        return None
    return value


def int_or_intlist(value: str):
    if value is None:
        raise ValueError("indices_unit_pert must not be None")

    value = value.strip()
    if not value:
        raise ValueError("indices_unit_pert cannot be an empty string")

    if ',' in value:
        return [int(x.strip()) for x in value.split(',')]
    return int(value)


# =============================================================================
def read_input_parameters(parfile_path, log_run, par=None):
    """
    Read input parameters from Parfile.
    """

    if par is None: 

        config = configparser.ConfigParser(
                                            converters={'maybeint': maybeint, 
                                                        'intlist': int_or_intlist,
                                                        'maybestr': maybestr}
                                            )
        if len(config.read(parfile_path)) == 0:
            raise ValueError("Failed to open/find a parameters file!")

        par = InputParameters()
        log_run.info("PARFILE")

        def log_config_section(log_run, config, section):
            log_run.info(f"\n------ {section} ------")
            for key, value in config.items(section):
                log_run.info(f"{key} = {value}")

        # ---- FilePaths ----
        section = 'FilePaths'
        log_config_section(log_run, config, section)
        par.model_filename = config.get(section, 'model_filename', fallback=par.model_filename)
        par.perturbation_filename = config.get(section, 'perturbation_filename', fallback=par.perturbation_filename)
        par.local_weights_filename = config.get(section, 'local_weights_filename', fallback=par.local_weights_filename)
        par.mask_filename = config.get(section, 'mask_filename', fallback=par.mask_filename)
        par.data_vals_filename = config.get(section, 'data_vals_filename', fallback=par.data_vals_filename)
        par.data_background_filename = config.get(section, 'data_background_filename', fallback=par.data_background_filename)
        par.path_output = config.get(section, 'path_output', fallback=par.path_output)
        par.sensit_path = config.get(section, 'sensit_path', fallback=par.sensit_path)

        par.model_filename = parse_value(par.model_filename)
        par.perturbation_filename = parse_value(par.perturbation_filename)
        par.local_weights_filename = parse_value(par.local_weights_filename)
        par.mask_filename = parse_value(par.mask_filename)
        par.data_vals_filename = parse_value(par.data_vals_filename)
        par.data_background_filename = parse_value(par.data_background_filename)
        par.path_output = parse_value(par.path_output)
        par.sensit_path = parse_value(par.sensit_path)

        # ---- SolverParameters ----
        section = 'SolverParameters'
        log_config_section(log_run, config, section)
        par.sensit_type = config.get(section, 'sensit_type', fallback=par.sensit_type)
        par.unit_conv = config.getboolean(section, 'unit_conv', fallback=par.unit_conv)
        par.use_mask_domain = config.getboolean(section, 'use_mask_domain', fallback=par.use_mask_domain)
        par.num_epochs = config.getint(section, 'num_epochs', fallback=par.num_epochs)
        par.use_loaded_mask = config.getboolean(section, 'use_loaded_mask', fallback=par.use_loaded_mask)

        # ---- PreProcessingParameters ----
        section = 'PreProcessingParameters'
        log_config_section(log_run, config, section)
        par.ind_unit_mask = config.getint(section, 'ind_unit_mask', fallback=par.ind_unit_mask)
        par.distance_max = config.getint(section, 'distance_max', fallback=par.distance_max)

        # ---- SaveOutput ----  # TODO: path_output goes into SavePars class but is not raed here. Make this more coherent. 
        section = 'SaveOutput'
        log_config_section(log_run, config, section)
        par.save_plots = config.getboolean(section, 'save_plots', fallback=par.save_plots)
        par.save_interval  = config.getint(section, 'save_interval', fallback=par.save_interval )

        # ---- SamplingParams ----
        section = 'SamplingParams'
        log_config_section(log_run, config, section)
        
        par.indices_unit_pert = config.getintlist(section, 'indices_unit_pert', fallback=par.indices_unit_pert)
        par.ind_unit_force = config.getmaybeint(section, 'ind_unit_force', fallback=par.ind_unit_force)
        par.ind_unit_ref = config.getmaybeint(section, 'ind_unit_ref', fallback=par.ind_unit_ref)
        par.force_pert_type = config.getmaybestr(section, 'force_pert_type', fallback=par.force_pert_type)
        par.n_births_max = config.getint(section, 'n_births_max', fallback=par.n_births_max)
        par.use_dynamic_mask = config.getboolean(section, 'use_dynamic_mask', fallback=par.use_dynamic_mask)
        par.std_data_fit = config.getfloat(section, 'std_data_fit', fallback=par.std_data_fit)
        par.std_petro = config.getfloat(section, 'std_petro', fallback=par.std_petro)
        par.std_geom_glob = config.getfloat(section, 'std_geom_glob', fallback=par.std_geom_glob)
        par.force_pert_0 = config.getfloat(section, 'force_pert_0', fallback=par.force_pert_0)
        par.force_pert_1 = config.getfloat(section, 'force_pert_1', fallback=par.force_pert_1)
        par.force_pert_2 = config.getfloat(section, 'force_pert_2', fallback=par.force_pert_2)
        par.force_pert_3 = config.getfloat(section, 'force_pert_3', fallback=par.force_pert_3)
        par.force_pert_4 = config.getfloat(section, 'force_pert_4', fallback=par.force_pert_4)


        # ---- NoiseParams ----
        section = 'NoiseParams'
        log_config_section(log_run, config, section)
        par.factor_spectrum_min = config.getfloat(section, 'factor_spectrum_min', fallback=par.factor_spectrum_min)
        par.factor_spectrum_max = config.getfloat(section, 'factor_spectrum_max', fallback=par.factor_spectrum_max)
        par.amplitude_pert_min = config.getfloat(section, 'amplitude_pert_min', fallback=par.amplitude_pert_min)
        par.amplitude_pert_max = config.getfloat(section, 'amplitude_pert_max', fallback=par.amplitude_pert_max)
        par.weights = config.get(section, 'weights', fallback=par.weights)
        par.normalise = config.getboolean(section, 'normalise', fallback=par.normalise)

        par.correlation_length_0 = config.getfloat(section, 'correlation_length_0', fallback=par.correlation_length_0)
        par.correlation_length_1 = config.getfloat(section, 'correlation_length_1', fallback=par.correlation_length_1)
        par.correlation_length_2 = config.getfloat(section, 'correlation_length_2', fallback=par.correlation_length_2)

        par.corr_zx = config.getfloat(section, 'corr_zx', fallback=par.corr_zx)
        par.corr_zy = config.getfloat(section, 'corr_zy', fallback=par.corr_zy)
        par.corr_xy = config.getfloat(section, 'corr_xy', fallback=par.corr_xy)

        par.rotation_angle_0 = config.getfloat(section, 'rotation_angle_0', fallback=par.rotation_angle_0)
        par.rotation_angle_1 = config.getfloat(section, 'rotation_angle_1', fallback=par.rotation_angle_1)
        par.rotation_angle_2 = config.getfloat(section, 'rotation_angle_2', fallback=par.rotation_angle_2)

        log_run.info("Done reading input parameters. \n")

    else: 
        log_run.info("Not reading parfile. REUSE provided parameters! \n")

    return par
