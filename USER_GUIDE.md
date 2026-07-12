# User Guide: Trans-Dimensional Geometric Inversion

This guide helps users who have the code installed and want to run inversions on their own data.

## Quick Start

Run an inversion from the command line (from the repository root):

```bash
python main.py parfiles/parfile_transd_synth1.txt 42
```

where `parfiles/parfile_transd_synth1.txt` is the parameter file and `42` is a random seed for reproducibility.

Optional flags:

- `--no-logging` — disable logging entirely
- `--plot` — display and save metrics plots once the inversion completes


## What You Need

Before running, prepare the following files:

1. **Model file** — A Tomofast-x format file defining your 3D model grid and initial property values (typically density in kg/m³)
2. **Data file** — Observed gravity or magnetic data
3. **Sensitivity matrix** — Pre-computed sensitivity kernel from Tomofast-x (stored as multiple files in a folder)
4. **Parameter file** — Configuration file controlling the inversion (see below)


## Understanding the Output

Results are written to the folder specified by `path_output`. Model files are only written when `save_plots = True`, and only every `save_interval` iterations; the metrics files below are always written.

| File | Description |
|------|-------------|
| `m_curr*.vts` | Accepted 3D models at saved iterations (structured grid, viewable in ParaView) |
| `data_calc_*.vtp` | Calculated data response at saved iterations |
| `data_residuals_*.vtp` | Data residuals (observed minus calculated) at saved iterations |
| `checkpoint_latest.pkl` / `checkpoint_*.pkl` | Restart checkpoints (written every `checkpoint_interval` iterations) |
| `metrics_summary.txt`, `metrics_data.csv` | Convergence metrics for the run |
| `metrics_plot.png` | Metrics plot (only when `--plot` is used) |
| `parfile_*.txt` | Copy of your parameter file (for reproducibility) |

The log file `log_file.log` is written to the directory you launch the command from (the working directory), not to `path_output`.


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
| `unit_conv` | Apply a 1e2 factor to convert gravity data to mGal (`True`/`False`) — set `False` for magnetic data |
| `use_mask_domain` | Restrict perturbations using a domain mask (`True`/`False`) |
| `num_epochs` | Number of MCMC iterations (model proposals) |
| `use_loaded_mask` | Load mask from file instead of generating one (`True`/`False`) |

### [PreProcessingParameters]

| Parameter | Description |
|-----------|-------------|
| `ind_unit_mask` | Index of the rock unit (ordered by increasing density) used to define the perturbation mask |
| `distance_max` | Maximum distance, in number of cells, from that unit's outline within which perturbations are allowed |

### [SaveOutput]

| Parameter | Description |
|-----------|-------------|
| `save_plots` | Whether models are written to disk (`True`/`False`) |
| `save_interval` | Number of iterations between saved models (`1` saves every model) |
| `checkpoint_interval` | Number of iterations between restart checkpoints (optional; defaults if omitted) |

### [SamplingParams]

| Parameter | Description |
|-----------|-------------|
| `indices_unit_pert` | Comma-separated indices of units to perturb (e.g., `0,1,2`), or `all` |
| `ind_unit_force` | Index of unit for forced perturbations (`None` to disable) |
| `ind_unit_ref` | Index of reference unit for guided changes (`None` to disable) |
| `n_births_max` | Maximum number of new units that can be born |
| `force_pert_type` | Type of forced perturbation: `petrophy_increase`, `petrophy_decrease`, `geometry`, or `None` |
| `use_dynamic_mask` | Update perturbation mask during sampling (`True`/`False`) |
| `std_data_fit` | Standard deviation of the data likelihood. Larger values loosen the data fit (more proposals accepted); smaller values tighten it (fewer accepted) |
| `std_petro` | Standard deviation for petrophysical values/perturbations |
| `std_geom_glob` | Weight of the prior (geometric) model term in the cost function |
| `force_pert_0` | Probability of forcing acceptance for type 0 (user-defined) perturbations |
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
| `weights` | Optional local weights multiplying the noise values (can act as a mask); `None` for uniform |
| `normalise` | Normalise noise values before scaling (`True`/`False`) |
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
| 3 | Birth | Add a new geological unit where the data-misfit gradient is high |
| 4 | Death | Remove an existing unit |

For each proposal, the algorithm computes the forward response and evaluates a Metropolis-Hastings acceptance ratio combining the data likelihood and the prior terms. Each move type can additionally be force-accepted with a user-set probability (the `force_pert_*` parameters), which breaks detailed balance and makes the chain non-reversible, favouring broad exploration over a calibrated posterior sample. The model dimension can change during sampling (trans-dimensional), allowing the number of geological units to be driven by the data.

### Level Set Parameterization

Unlike traditional cell-based inversions, this framework uses **signed distance functions** to represent unit boundaries. Perturbations modify these distance fields, causing boundaries to expand or contract smoothly. This approach:

- Produces geologically plausible structures with smooth boundaries
- Naturally handles topology changes (units merging or splitting)
- Allows efficient birth/death of geological units


## Tips for New Users

1. **Start with the example** — Run the provided synthetic example to understand the workflow before using your own data

2. **Check your sensitivity matrix** — Ensure it matches your model grid dimensions

3. **Start with few epochs** — Use `num_epochs = 20-50` initially to check that everything works before running longer chains

4. **Tune acceptance rates** — If models rarely change, increase `std_data_fit`; if they change too chaotically, decrease it

5. **View results in ParaView** — Open the `.vts` and `.vtp` files to visualize how the model evolves


## Reproducibility

The code automatically saves the random seed (in the log) and a copy of your parameter file to the output directory. To reproduce a run exactly, use the same seed and parameter file.
