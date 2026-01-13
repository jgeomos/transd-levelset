# User Guide: Trans-Dimensional Geometric Inversion

This guide helps users who have the code installed and want to run inversions on their own data.

## Quick Start

Run an inversion from the command line:

```bash
python main.py parfile_transd_synth1.txt 42
```

where, for instance, `parfile_transd_synth1.txt` is the parameter file and `42` is a random seed for reproducibility. Use `--no-logging` to disable console output.


## What You Need

Before running, prepare the following files:

1. **Model file** — A Tomofast-x format file defining your 3D model grid and initial property values (typically density in kg/m³)
2. **Data file** — Observed gravity or magnetic data
3. **Sensitivity matrix** — Pre-computed sensitivity kernel from Tomofast-x (stored as multiple files in a folder)
4. **Parameter file** — Configuration file controlling the inversion (see below)


## Understanding the Output

Results are saved to the folder specified by `path_output`. Key outputs include:

| File | Description |
|------|-------------|
| `*.vtk` | 3D models at various iterations (viewable in ParaView) |
| `transd_inversion.log` | Full log of the inversion process |
| `parfile_*.txt` | Copy of your parameter file (for reproducibility) |


## Parameter File Reference

The parameter file is organized into sections. Below is a description of each parameter.

### [FilePaths]

| Parameter | Description |
|-----------|-------------|
| `model_filename` | Path to starting model (Tomofast-x format) |
| `perturbation_filename` | Optional initial perturbation to add to model (`None` to skip) |
| `mask_filename` | Optional mask file restricting where changes occur (`None` for no mask) |
| `local_weights_filename` | Optional weights for prior model term (`None` for uniform) |
| `data_vals_filename` | Path to observed gravity/magnetic data |
| `data_background_filename` | Optional background response to subtract (`None` to skip) |
| `path_output` | Output directory for results |
| `sensit_path` | Directory containing sensitivity matrix files |

### [SolverParameters]

| Parameter | Description |
|-----------|-------------|
| `sensit_type` | Type of data: `grav` (gravity) or `magn` (magnetic) |
| `unit_conv` | Convert gravity units to mGal (`True`/`False`) — set `False` for magnetic data |
| `use_mask_domain` | Restrict perturbations using a domain mask (`True`/`False`) |
| `num_epochs` | Number of MCMC iterations (model proposals) |
| `use_loaded_mask` | Load mask from file instead of generating one (`True`/`False`) |

### [SamplingParams]

| Parameter | Description |
|-----------|-------------|
| `indices_unit_pert` | Comma-separated indices of units to perturb (e.g., `0,1,2`) |
| `ind_unit_force` | Index of unit for forced perturbations (`None` to disable) |
| `ind_unit_ref` | Index of reference unit for guided changes (`None` to disable) |
| `n_births_max` | Maximum number of new units that can be born |
| `force_pert_type` | Type of forced perturbation: `petrophy_increase`, `petrophy_decrease`, `geometry`, or `None` |
| `use_dynamic_mask` | Update perturbation mask during sampling (`True`/`False`) |
| `std_data_fit` | Standard deviation for data likelihood (controls data fit weight) |
| `std_petro` | Standard deviation for petrophysical perturbations |
| `std_geom_glob` | Weight of prior model term in cost function |
| `force_pert_0` | Probability of forcing acceptance for type 0 (forced) perturbations |
| `force_pert_1` | Probability of forcing acceptance for type 1 (geometric) perturbations |
| `force_pert_2` | Probability of forcing acceptance for type 2 (petrophysical) perturbations |
| `force_pert_3` | Probability of forcing acceptance for type 3 (birth) moves |
| `force_pert_4` | Probability of forcing acceptance for type 4 (death) moves |

### [NoiseParams] — Random Field Generation

These parameters control the correlated random fields used for geometric perturbations.

| Parameter | Description |
|-----------|-------------|
| `factor_spectrum_min/max` | Range for spectral power factor (controls blob size) |
| `amplitude_pert_min/max` | Range for perturbation amplitude |
| `normalise` | Normalize noise values before scaling (`True`/`False`) |
| `correlation_length_0/1/2` | Correlation lengths along z, x, y axes |
| `corr_zx`, `corr_zy`, `corr_xy` | Cross-correlation between dimensions (0 to 1) |
| `rotation_angle_0/1/2` | Rotation angles around z, x, y axes (degrees) |


## Technical Overview

### What the Code Does

This framework performs **trans-dimensional Bayesian inversion** of gravity or magnetic data to infer subsurface geological structures. The key distinguishing feature is its **geometric approach** using level set methods to represent unit boundaries.

### The Algorithm

At each iteration, the sampler randomly selects one of five perturbation types:

| Type | Name | Description |
|------|------|-------------|
| 0 | Forced | Apply pre-defined changes (optional, for guided inversion) |
| 1 | Geometric | Modify unit boundaries using correlated random fields |
| 2 | Petrophysical | Change property values (density/susceptibility) within units |
| 3 | Birth | Add a new geological unit at high-gradient locations |
| 4 | Death | Remove an existing unit |

For each proposal, the algorithm computes the forward response and uses a **Metropolis-Hastings acceptance criterion** based on data fit and prior constraints. The model dimension can change during sampling (trans-dimensional), allowing the number of geological units to be determined by the data.

### Level Set Parameterization

Unlike traditional cell-based inversions, this framework uses **signed distance functions** to represent unit boundaries. Perturbations modify these distance fields, causing boundaries to expand or contract smoothly. This approach:

- Produces geologically plausible structures with smooth boundaries
- Naturally handles topology changes (units merging or splitting)
- Allows efficient birth/death of geological units


## Tips for New Users

1. **Start with the example** — Run the provided synthetic example to understand the workflow before using your own data
2. **Start with few epochs** — Use `num_epochs = 20-50` initially to check that everything works before running longer chains
3. **Tune acceptance rates** — If models rarely change, decrease `std_data_fit`; if they change too chaotically, increase it
4. **View results in ParaView** — Open the `.vtk` files to visualize how the model evolves, and analyse plots produced by the Python scripts to assess convervence or make a diagnonis on potential issues.


## Reproducibility

The code automatically saves the random seed and a copy of your parameter file to the output directory. To reproduce a run exactly, use the same seed and parameter file.
