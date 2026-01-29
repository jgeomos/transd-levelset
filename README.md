# Geometrical trans-dimensional Bayesian inversion of potential fields data

## A Python code for trans-dimensional geometrical inversion of potential field data using level-set geometry and birth–death sampling.

This implementation is based on and builds upon the method proposed in the following paper: <br>
Giraud, J., Ogarko, V., Caumon, G., Pirot, G., Portes dos Santos, L., Cupillard, P., & Herrero, J. (2025). Pseudo trans-dimensional 3D geometrical inversion: A proof of concept using gravity data. (Geophysical Journal International, Accepted, in press)

Results might differ slightly due to the stochastic nature of the sampling process and modifications of the original scripts. 
Alpha version with new functionalities.

Code authors:<br/>
Jeremie Giraud<sup>1,2</sup>, Vitaliy Ogarko<sup>1,3</sup><br/>
<sup>1</sup> Centre for Exploration Targeting, School of Earth and Oceans, The University of Western Australia (Perth, Australia)<br/>
<sup>2</sup> RING Team, GeoRessources, Universite de Lorraine (Nancy, France)<br/>
<sup>3</sup> Mineral Exploration Cooperative Research Centre, The University of Western Australia (Perth, Australia) 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Overview 

This project provides a Python framework for performing Bayesian trans-dimensional inversion of gravity and magnetic data to recover 3D geological structures. The code explores the model space with the generation of models with varying numbers of geological units, enabling data-driven determination of model complexity, together with variable petrophysical properties and rock unit geometry.
The process relies on a Markov Chain Monte Carlo (MCMC) sampling with level set representations, in a non-revertible Metropolis-Hastings algorithm. 

**Key capabilities:**
- Trans-dimensional sampling (variable number of units)
- Level set representation of geological units
- Birth/death moves for units
- Nullspace exploration with constraints
- Efficient forward modeling with Tomofast-x integration
- Real-time convergence monitoring

## Features

### Sampling Methods
- **Geometric perturbations**: Modify unit boundaries via correlated random fields with user-defined properties.
- **Petrophysical perturbations**: Update density/susceptibility values sampling from Gaussian distributions or a user-defined uniform, defining bounds. 
- **Birth moves**: Nucleate new geological units (default: conforming to existing units)
- **Death moves**: Remove existing units (default: killing smaller units first)
- **Forced perturbations**: Impose user-defined structural or petrophysical changes

### Forward Modeling
- Integration with Tomofast-x sensitivity matrices: the code uses sensitivity matrices generated using the Tomofast-x platform
- Support for gravity and magnetic data

### Visualization
- Models saved with VTK (*.vts files)
- Model evolution with GIF animations in the notebook.
- Cross-section viewers 
- Convergence diagnostics
- Model flythrough animations 

## Installation 
`pip install -r requirements.txt` to install the required packages.<br>
or<br>
`pip install numpy scipy numba scikit-fmm cc3d matplotlib vtk`<br>
`pip install colorcet`

## Quick Start
### 1. Run example
`python main.py ./parfiles/parfile_transd_synth1.txt 42`<br>
Runs the trans-dimensional inversion using the specified parameter file (`-p`) and random seed (`-s`) for reproducibility.
or using the notebook `transd_main.ipynb`
### 2. For your data: Prepare your input files (see Input Section below). 
- a parameter file (e.g., <"your_parfile_transd.txt">) containing the parameters for the inversion as well as path to input files containing data and models,
- data file,
- prior model file ,
- sensitivity matrix,
- optional: mask file, 
- optional: spatially varying confidence values in prior model.

Data and model structure follow the same formats as Tomofast-x's. 
Input parameters are set in the parameter file (*.txt) and can be set manually by the User. 
Advanced parameters are hard-coded in dedicated classes in Python files. 

To run inversion, adjust the parameter file and run using 1., specifying your parfile name.

For detailed parameter descriptions and usage tips, see [USER_GUIDE.md](USER_GUIDE.md).

### Inputs
Data and model files follow Tomofast-X format:
- **Model grid**: ASCII file with cell coordinates and initial properties
- **Geophysical data**: Observation locations and values
- **Sensitivity matrix**: Pre-computed forward operator (calculated using Tomofast-x, using the parameter file in ./parfiles/tomofast_parfiles/)
- **Parameter file**: Text file controlling inversion settings

### Outputs
- **VTK models** (`m_curr_*.vts`): 3D models at each accepted iteration
- **Metrics** (`metrics.csv`): Misfit evolution, acceptance rates
- **Acceptance history** (`accepted_changes.json`): Detailed move tracking

#### Model Files
- `m_curr_*.vts`: VTK structured grid files (ParaView compatible)
- `metrics.csv`: Convergence metrics (misfit, acceptance rate)
- `accepted_changes.txt`: History of accepted moves

#### Visualizations
- `inversion_animation.gif`: Model evolution, saved to disk. 
- `convergence_plot.png`: Misfit vs iteration, saved to disk. 
- `acceptance_history.png`: Move type acceptance, saved to disk. 

### Analysis
Dedicated worflow using Haralick features and t-SNE (see Giraud et al. 2025 and references therein)
- Posterior model statistics
- Uncertainty quantification

## Project Structure
src/ contains modules that are necessary to run inversion or to view results.
```
├── transd_main.ipynb              # Jupyter notebook to run inversion with plots.
├── main.py                        # Script to run invesion (no plots). 
├── src/
│   ├── transd_runner.py           # Orchestrate the execution
│   ├── transd_solver.py        # MCMC sampling engine
│   ├── model_state.py             # Model state manager
│   ├── petro_state.py             # Petrophysical state manager
│   ├── birth_death.py             # Birth/death operations
│   ├── inversion_metrics.py       # Convergence, acceptance ratio, petrophysical and model tracking
│   │
│   ├── forward_solver/            # Forward modeling
│   │   ├── forward_calculation.py # Calculate the forward data
│   │   └── tomofast_sensit.py     # Class and units for handling of Tomofast-x sensitivity matrices
│   │
│   ├── random_gen/                # Perturbation generators
│   │   ├── scalar_sampling.py     # Generation of random numbers for perturbation selection and petrophysics
│   │   └── anisotropic_noise.py   # Generation of anisotropic random fields for perturbation of level-sets
│   │
│   ├── input_output/              # Data I/O
│   │   ├── input_params.py        # Contains function to parse inputs from parfile
│   │   ├── output_manager.py      # For management of outputs 
│   │   └── tomofast_reading_utils.py # To read files using Tomofast-x's format
│   │
│   └── utils/                     # Utilities
│       ├── plot_utils.py          # Plotting and saving GIF
│       └── quality_control.py     # Errors and main holder for inversion consistency checks
│       └── transd_utils.py        # Functions to carry out computation used in the solver. 
│
├── parfiles/                      # Parameter files controlling the inversion/nullspace navigation
│   ├── tomofast_parfiles/         # Parfiles used for the calculation of the sensitivity matrix with Tomofast.
└── data/                          # Contains the data and model for inversion
```

## Common Issues
### Install 
VTK is the package that can cause issues when installing. 
Input files not in the proper folder.

### Prior to running
Make sure all files are in the proper folder!

## Citation

Associated to the publication (accepted but not published):<br>
Giraud, J., Ogarko, V., Caumon, G., Pirot, G., Portes dos Santos, L., Cupillard, P., & Herrero, J. (2025). Pseudo trans-dimensional 3D geometrical inversion: A proof of concept using gravity data. (Accepted, in press)

## Contributing

Contributions are welcome! 
Please open an issue or submit a pull request. 

## Ideas, wishlist, possible future dev.:
- short term: create a module with array operations e.g array_utils.py, or, more focussed, pure numpy operations, e.g. array_ops.py, with only scipy and numpy dependency for use in other module. 

## Acknowledgments
Jeremie Giraud acknowledges support from European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no. 101032994 and from the Australian Research Council through the Early Career Industry Fellowship grant no. IE240100281.

## Contact
Jeremie Giraud, jeremie.giraud@uwa.edu.au

## Related Projects

- [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x): 3D parallel inversion
- [GeoMos nullspace](https://github.com/jgeomos/GeoMos-nullspace): null-space navigation

## Relevant references 

For a case study, refer to Giraud et al 2024b.

Bodin, T., & Sambridge, M. (2009). *Seismic tomography with the reversible jump algorithm.* **Geophysical Journal International**, 178, 1411–1436.

Giraud, J., Ford, M., Caumon, G., Ogarko, V., Grose, L., Martin, R., & Cupillard, P. (2024a). *Geologically constrained geometry inversion and null-space navigation to explore alternative geological scenarios: A case study in the Western Pyrenees.* **Geophysical Journal International**, 239, 1359–1379.

Giraud, J., Rashidifard, M., Ogarko, V., Caumon, G., Grose, L., Herrero, J., Cupillard, P., Lindsay, M., Jessell, M., & Aillères, L. (2024b). *Transdimensional geometrical inversion: Application to undercover imaging using gravity data.* **SEG Global Meeting Abstracts**, 167–170.

Green, P. J. (1995). *Reversible Jump Markov Chain Monte Carlo Computation and Bayesian Model Determination.* **Biometrika**, 82, 711.

Mosegaard, K., & Tarantola, A. (1995). *Monte Carlo sampling of solutions to inverse problems.* **Journal of Geophysical Research: Solid Earth**, 100, 12431–12447.

Sambridge, M., Gallagher, K., Jackson, A., & Rickwood, P. (2006). *Trans-dimensional inverse problems, model comparison and the evidence.* **Geophysical Journal International**, 167, 528–542.
