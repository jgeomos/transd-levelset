"""
Visualization and plotting utilities for inversion results.

Provides comprehensive plotting functions for monitoring and analyzing
trans-dimensional inversion outputs, including convergence diagnostics,
model evolution animations, and quality control visualizations.

Main Components
---------------
PlotParameters : dataclass
    Configuration for plot appearance and slicing
plot_metrics : function
    Visualize convergence metrics (misfit, acceptance rates, prior costs)
create_inversion_animation : function
    Generate GIF animations of model evolution
plot_model_slice : function
    Display 2D slices through 3D density/susceptibility models

Notes
-----
- Uses VTK for reading 3D model files (.vts format)
- Generates publication-quality figures with matplotlib
- Supports custom colormaps via colorcet (optional dependency)

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

from dataclasses import dataclass
import matplotlib.pylab as plt
import numpy as np
import colorcet as cc  # Used only for colormaps.
import random as rd
from typing import Optional
import os
import vtk

from vtk.util.numpy_support import vtk_to_numpy
import re
import glob
from PIL import Image
import matplotlib.gridspec as gridspec

from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

plt.ioff()

@dataclass(slots=True)
class PlotParameters:
    """
    Parameters for plots of null space navigation outputs.
    at the moment: plots only in the x direction.
    """

    # Threshold for identification of density differences between models before/after navigation.
    dens_tresh: float = 30
    # Value of density contrast defining range of colors min and max.
    colm: float = 250
    # Slices for plots.
    slice_x: int = -1
    slice_y: int = -1
    slice_z: int = -1
    plot_models: Optional[tuple] = None
    # Titles for plots.
    plot_titles: Optional[tuple] = None
    # Titles for color bars.
    cbar_titles: Optional[tuple] = None
    # Limits for colours on plot.
    clims: Optional[tuple] = None
    # Colour schemes for plots.
    colorschemes: Optional[tuple] = None
    # Ticks for colorbar.
    cbar_ticks: Optional[tuple] = None
    # limits in x and y directions for plots
    xlims: Optional[np.array] = None
    ylims: Optional[np.array] = None


def plot_metrics(metrics, run_params, data_misfit_lims=np.array([0.25, 5.0]), print_stats=True, 
                    show=False, save=True, plot_path=None):
    """
    Plot monitoring metrics including acceptance types.
    
    Parameters
    ----------
    metrics : MonitoringMetrics
        Metrics object with tracking data
    run_params : RunBaseParams
        Run parameters with perturbation type names
    data_misfit_lims : array-like, optional
        Y-axis limits for data misfit plot
    """
    metrics.it_accepted_model = np.array(metrics.it_accepted_model)
    metrics.it_accepted_type = np.array(metrics.it_accepted_type)
    
    # Create figure with 4 rows, 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(12, 12))
    
    # Data misfit
    ax = axes[0, 0]
    ax.plot(metrics.data_misfit, label='All proposals')
    ax.plot(metrics.it_accepted_model, metrics.data_misfit[metrics.it_accepted_model], 
            'o', label='Accepted', markersize=4)
    ax.set_ylim(data_misfit_lims)
    ax.set_title('Data Misfit')
    ax.set_xlabel('Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log likelihood ratio
    ax = axes[0, 1]
    ax.plot(metrics.log_likelihood_ratio)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Log Likelihood Ratio')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    # Accept ratio
    ax = axes[1, 0]
    # ax.plot(metrics.log_priorgeom_ratio)
    ax.plot(metrics.model_misfit)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    # ax.set_title('Log Prior (Geometry) Ratio')
    ax.set_title('Sign Dist Model Misfit')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    # Metrics on geometry.
    ax = axes[1, 1]
    # ax.plot(metrics.log_priorgeom_ratio)
    ax.plot(metrics.log_priorgeom_ratio)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    # ax.set_title('Log Prior (Geometry) Ratio')
    ax.set_title('Log prior geom model Ratio')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    # Petro misfit
    ax = axes[2, 0]
    ax.plot(metrics.petro_misfit)
    ax.set_title('Petrophysical Misfit')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    # Log prior petro ratio
    ax = axes[2, 1]
    ax.plot(metrics.log_priorpetro_ratio)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Log Prior (Petrophy) Ratio')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)
    
    ax = axes[3, 0] 
    
    # Define colors and markers for each perturbation type
    type_styles = {
        0: {'color': '#1f77b4', 'marker': 's', 'size': 6},  # Blue square - forced
        1: {'color': '#ff7f0e', 'marker': 'o', 'size': 5},  # Orange circle - geometry
        2: {'color': '#2ca02c', 'marker': '^', 'size': 6},  # Green triangle - petrophy
        3: {'color': '#d62728', 'marker': '*', 'size': 9},  # Red star - birth
        4: {'color': '#9467bd', 'marker': 'X', 'size': 7}   # Purple X - death
    }
    
    # Extract short names from run_params messages
    type_labels = {}
    for pert_type in range(5):
        msg = run_params.perturbation_messages.get(pert_type, f"Type {pert_type}")
        if "->" in msg:
            short_name = msg.split("->")[-1].strip()
            short_name = short_name.replace("perturbation", "").replace("Random ", "").strip()
        else:
            short_name = msg
        type_labels[pert_type] = short_name
    
    # Plot all data misfit values
    ax.plot(metrics.data_misfit, label='All proposals')
    
    # Plot accepted points by type
    plotted_types = set()
    for i in range(len(metrics.it_accepted_model)):
        it = metrics.it_accepted_model[i]
        ptype = metrics.it_accepted_type[i]
        style = type_styles.get(ptype, {'color': 'gray', 'marker': 'o', 'size': 5})
        label = f"{ptype}: {type_labels[ptype]}" if ptype not in plotted_types else None
        ax.plot(it, metrics.data_misfit[it], 
                marker=style['marker'], 
                color=style['color'], 
                markersize=style['size'],
                linestyle='none',
                label=label)
        plotted_types.add(ptype)
    
    ax.set_ylim(data_misfit_lims)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Data Misfit')
    ax.set_title('Data Misfit by Accepted Perturbation Type')
    ax.legend(loc='best', fontsize=9)
    # ax.grid(True, alpha=0.3)

    ax = axes[3, 1] 
    ax.plot(metrics.accept_ratio)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Acceptance Ratio')
    ax.set_xlabel('Iteration')
    ax.grid(True, alpha=0.3)

    ax = axes[4, 0] 
    ax.plot(metrics.n_units_total, label='Number of units')
    ax.set_title('Number of units during modelling')
    ax.set_xlabel('Iteration')

    ax = axes[4, 1]
    ax.plot(metrics.log_posterior, label='log-posterior')
    ax.set_title('Evolution of log-posterior')
    ax.set_xlabel('Iteration')

    for ax in axes.flat:
        add_grid(ax)

    # Print statistics
    if print_stats:
        acceptance_rate = len(metrics.data_misfit[metrics.it_accepted_model]) / len(metrics.data_misfit)
        print(f"\nAcceptance rate: {acceptance_rate:.3f}")
        print(f"Mean accept ratio: {np.mean(metrics.accept_ratio):.3f}")
        
        # Print breakdown by type
        print("\nAccepted changes by type:")
        for pert_type in range(5):
            count = metrics.count_by_type(pert_type)
            if count > 0:
                print(f"  {type_labels[pert_type]}: {count} ({count/len(metrics.it_accepted_model)*100:.1f}%)")
    
    plt.tight_layout()

    if save:
        if plot_path is None: 
            raise ValueError(f"{save} is set to True but {plot_path} is None!")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved in {plot_path}')
        if show:
            plt.show()

    if show: 
        plt.show()

    return None
    

def create_inversion_animation(metrics, gpars, output_dir, 
                               filename='inversion_animation.gif',
                               axis='y', slice_index=None,
                               fps=200, 
                               clim=np.array((2500, 3250)),
                               padding_start=9, 
                               padding_end=-10,
                               decimate=4, 
                               dpi=150,
                               filename_pattern='m_curr_',
                               max_iteration=None):
    """
    Create animated GIF showing model evolution during inversion.
    
    Reads saved VTS model files and creates an animation showing model sections
    alongside misfit evolution and acceptance rate.
    
    Parameters
    ----------
    metrics : InversionMetrics
        Metrics object containing iteration history and misfit values
    gpars : GridParameters
        Grid parameters with model dimensions
    output_dir : str or Path
        Directory containing saved VTS files
    filename : str, default='inversion_animation.gif'
        Output filename for the GIF
    axis : str, default='y'
        Slice axis ('x', 'y', or 'z')
    slice_index : int, optional
        Index of slice to show. If None, uses middle slice
    fps : float, default=200
        Frames per second (higher = slower animation)
    padding_start : int, default=9
        Number of iterations to skip at start
    padding_end : int, default=-10
        Number of iterations to skip at end (negative counts from end)
    decimate : int, default=4
        Show every Nth iteration (1 = all, 4 = every 4th)
    dpi : int, default=150
        Resolution of output images
    filename_pattern : str, default='m_curr_'
        Pattern for matching VTS files
    max_iteration : int, optional
        Maximum iteration to show. If None, uses all accepted iterations
        
    Returns
    -------
    str
        Path to created GIF file
        
    Notes
    -----
    Only creates animation for accepted iterations stored in metrics.it_accepted_model.
    The animation shows three panels:
    - Model section at current iteration
    - Data misfit evolution
    - Acceptance ratio evolution
    
    Examples
    --------
    >>> # Basic usage - middle slice, default settings
    >>> gif_path = create_inversion_animation(metrics, gpars, 'output/')
    
    >>> # Custom slice and faster animation
    >>> gif_path = create_inversion_animation(
    ...     metrics, gpars, 'output/',
    ...     filename='my_animation.gif',
    ...     axis='y', slice_index=20,
    ...     fps=100, decimate=2
    ... )
    
    >>> # Display in notebook
    >>> from IPython.display import Image
    >>> Image(filename=gif_path)
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # Determine slice index if not provided
    if slice_index is None:
        axis_dims = {'x': gpars.dim[2], 'y': gpars.dim[1], 'z': gpars.dim[0]}
        slice_index = axis_dims[axis] // 2
        print(f"Using middle {axis}-slice: index {slice_index}")
    
    # Get accepted iterations and corresponding metrics
    x_iterations = np.array(metrics.it_accepted_model)
    y_misfit = metrics.data_misfit[x_iterations]
    accept_ratio = metrics.accept_ratio[x_iterations]
    
    # Determine maximum iteration to show
    if max_iteration is None:
        max_iteration = np.max(x_iterations)
    
    # Create the GIF
    print(f"Creating animation: {filename}")
    print(f"    Directory: {output_dir}")
    print(f"    Slice: {axis}={slice_index}")
    print(f"    Iterations: {len(x_iterations)} accepted")
    print(f"    Decimation: every {decimate} iterations")
    
    output_path = create_section_gif_from_vts(
        vts_folder=str(output_dir),
        file_name_gif=filename,
        padding_start=padding_start,
        padding_end=padding_end,
        x_misfit=x_iterations,
        y_misfit=y_misfit,
        max_x_misfit=max_iteration,
        accept_ratio=accept_ratio,
        gpars_class=gpars,
        axis=axis,
        index=slice_index,
        fps=fps,
        clim=clim,
        required_strings=filename_pattern,
        files=decimate,
        dpi=dpi
    )
    
    print(f"✓ Created: {output_path}")
    return output_path


def create_section_gif_from_vts(vts_folder, 
                                file_name_gif, 
                                padding_start, padding_end, 
                                x_misfit, y_misfit, 
                                max_x_misfit, 
                                accept_ratio, 
                                gpars_class, 
                                axis='z', 
                                index=None, 
                                fps=10, 
                                clim=np.array((2400, 3250)),
                                required_strings=None, 
                                files='All', 
                                dpi=150):
    """
    Create a GIF from VTS files in a folder, filtering by strings in filenames.

    Parameters:
    - vts_folder (str): Path to folder containing .vts files
    - file_name_gif (str): Output .gif path
    - axis (str): Section axis ('x', 'y', 'z')
    - index (int): Index to extract
    - duration (float): Duration of each frame in seconds
    - required_strings (list of str): Substrings that must all be in the filename
    """

    # Make file_name_gif contain the output folder. 
    file_name_gif = vts_folder + "\\" + file_name_gif

    # Gather all .vts files
    all_vts_files = glob.glob(os.path.join(vts_folder, '*.vts'))

    # Filter files based on required substrings
    if required_strings:
        filtered_files = [
            f for f in all_vts_files if all(substring in os.path.basename(f) for substring in required_strings)
        ]
    else:
        filtered_files = all_vts_files

    # Sort naturally (e.g. file1, file2, file10)
    if files=='All':
        vts_files = sorted(filtered_files, key=natural_sort_key)  # [::2]  # To do only 20 first plots [:20]
    else:
        vts_files = sorted(filtered_files[::files], key=natural_sort_key)  # [::2]  # To do only 20 first plots [:20]

    temp_dir = "temp_slice_images"
    os.makedirs(temp_dir, exist_ok=True)

    y_lims = np.zeros(2)
    y_lims[0] = np.min(y_misfit)
    y_lims[1] = np.max(y_misfit)

    image_files = []

    for i, vts_file in enumerate(vts_files):  # TODO: problem with i when getting value of x_misfit: this is the index of the read model, not hte model along the chain!!!
        print(f'Getting {vts_file}')
        slice_data1 = extract_slice_from_vts(vts_file, axis=axis, index=index)
        img_path = os.path.join(temp_dir, f"slice_{i:03d}.png")
        slice_data1 = slice_data1[::-1, :]
        if i==0:
            slice_data0 = slice_data1.copy()
        save_slice_as_image(slice_data0, slice_data1, 
                            padding_start, padding_end, 
                            x_misfit[:i], y_misfit[:i], y_lims, max_x_misfit, accept_ratio[:i], 
                            gpars_class, img_path, index, clim=clim, ind=len(x_misfit[:i]), dpi=dpi)
        image_files.append(img_path)
        print('Save as: ',  img_path)

    # Make the GIF
    images = [Image.open(img) for img in image_files]
    images[0].save(file_name_gif, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)

    print("GIF done with " + str(fps) + "FPS")
    print(f"GIF saved to {file_name_gif}")

    return file_name_gif


def create_model_flythrough_gif(model, gpars, output_path,
                                axis='x',
                                slice_range=None,
                                decimate=1,
                                fps=10,
                                dpi=150,
                                figsize=(10, 8),
                                cmap=cc.cm.CET_R4,
                                vmin=None, vmax=None,
                                title_prefix='Slice',
                                show_colorbar=True,
                                show_slice_number=True, 
                                progress_every=10):
    """
    Create animated GIF flying through slices of a 3D model.
    
    Creates an animation that shows sequential slices through a 3D model,
    effectively creating a "fly-through" visualization.
    
    Parameters
    ----------
    model : ndarray
        3D model array with shape (nz, ny, nx)
    gpars : GridParameters
        Grid parameters with dimensions and spacing
    output_path : str or Path
        Path for output GIF file (e.g., 'flythrough.gif')
    axis : str, default='x'
        Axis to slice along:
        - 'x': slices perpendicular to x-axis (yz planes)
        - 'y': slices perpendicular to y-axis (xz planes)
        - 'z': slices perpendicular to z-axis (xy planes)
    slice_range : tuple of int, optional
        (start, stop) slice indices. If None, uses all slices
    decimate : int, default=1
        Show every Nth slice (1=all, 2=every other, etc.)
    fps : int, default=10
        Frames per second (higher = faster animation)
    dpi : int, default=150
        Resolution of output images
    figsize : tuple, default=(10, 8)
        Figure size in inches (width, height)
    cmap : str, default='viridis'
        Matplotlib colormap name
    vmin, vmax : float, optional
        Color scale limits. If None, auto-scales to data range
    title_prefix : str, default='Slice'
        Prefix for slice titles
    show_colorbar : bool, default=True
        Whether to show colorbar
    show_slice_number : bool, default=True
        Whether to show slice number in title
        
    Returns
    -------
    str
        Path to created GIF file
        
    Examples
    --------
    >>> # Fly through all x-slices
    >>> gif_path = create_model_flythrough_gif(
    ...     mvars.m_curr, gpars, 'flythrough_x.gif', axis='x'
    ... )
    
    >>> # Faster animation, every 2nd slice
    >>> gif_path = create_model_flythrough_gif(
    ...     model, gpars, 'flythrough_fast.gif',
    ...     axis='y', decimate=2, fps=20
    ... )
    
    >>> # Custom range - middle section only
    >>> gif_path = create_model_flythrough_gif(
    ...     model, gpars, 'flythrough_middle.gif',
    ...     axis='z', slice_range=(10, 40)
    ... )
    
    >>> # Display in notebook
    >>> from IPython.display import Image
    >>> Image(filename=gif_path)
    """
    
    output_path = Path(output_path)
    
    # Ensure model has correct shape
    if model.shape != gpars.dim:
        model = model.reshape(gpars.dim)
    
    # Determine axis configuration
    # TODO: this should also be getting the values for the coordinates on horiz and vert axes
    axis_config = {
        'x': {
            'slice_dim': 2,
            'extent_dims': (1, 0),  # (y, z)
            'labels': ('Y', 'Z'),
            'max_slices': gpars.dim[2]
        },
        'y': {
            'slice_dim': 1,
            'extent_dims': (2, 0),  # (x, z)
            'labels': ('X', 'Z'),
            'max_slices': gpars.dim[1]
        },
        'z': {
            'slice_dim': 0,
            'extent_dims': (2, 1),  # (x, y)
            'labels': ('X', 'Y'),
            'max_slices': gpars.dim[0]
        }
    }
    
    if axis not in axis_config:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    
    config = axis_config[axis]
    slice_dim = config['slice_dim']
    max_slices = config['max_slices']
    xlabel, ylabel = config['labels']
    
    # Determine slice range
    if slice_range is None:
        start_slice, end_slice = 0, max_slices
    else:
        start_slice, end_slice = slice_range
        start_slice = max(0, start_slice)
        end_slice = min(max_slices, end_slice)
    
    # Generate slice indices
    slice_indices = np.arange(start_slice, end_slice, decimate)
    n_frames = len(slice_indices)
    
    print(f"Creating flythrough animation:")
    print(f"  Axis: {axis}")
    print(f"  Slices: {start_slice} to {end_slice} (every {decimate})")
    print(f"  Frames: {n_frames}")
    print(f"  Output: {output_path}")
    
    # Determine color scale
    if vmin is None:
        vmin = np.min(model)
    if vmax is None:
        vmax = np.max(model)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract first slice for initialization
    if axis == 'x':
        slice_data = model[:, :, slice_indices[0]].T
    elif axis == 'y':
        slice_data = model[:, slice_indices[0], :].T
    else:  # z
        slice_data = model[slice_indices[0], :, :]
    
    # Create initial image
    im = ax.imshow(slice_data.T, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', origin='lower', interpolation='nearest')
    
    # Set specific plot properties. 
    ax.set_aspect(1)
    ax.invert_yaxis()

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density (kg/m³)', rotation=270, labelpad=20)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Title
    if show_slice_number:
        title = ax.set_title(f'{title_prefix} {axis}={slice_indices[0]}', 
                           fontsize=14, fontweight='bold')
    else:
        title = ax.set_title(title_prefix, fontsize=14, fontweight='bold')
    
    def update(frame):
        """Update function for animation."""
        slice_idx = slice_indices[frame]
        
        # Extract slice
        if axis == 'x':
            slice_data = model[:, :, slice_idx]
        elif axis == 'y':
            slice_data = model[:, slice_idx, :]
        else:  # z
            slice_data = model[slice_idx, :, :]
        
        # Update image
        im.set_array(slice_data)
        
        # Update title
        if show_slice_number:
            title.set_text(f'{title_prefix} {axis}={slice_idx}')

        # Progress output
        if frame % progress_every == 0 or frame == n_frames - 1:
            print(f"  Frame {frame+1}/{n_frames} (slice {slice_idx})")
        
        return [im, title]
    
    # Create animation
    print("Generating frames...")
    anim = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save as GIF
    print("Saving GIF...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)

    plt.tight_layout()
    
    plt.close(fig)
    
    print(f"✓ Created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.2f} MB")
    
    return str(output_path)


def plot_model(ax, mesh_dim1, mesh_dim2, mod, title_string, cmap='viridis', clim=np.array((2600, 3290))):

    # color_min = clim[0]
    # color_max = clim[1]
    color_min = np.min(clim)
    color_max = np.max(clim)

    # TODO: make use of the PlotParameters class here

    handle = plt.pcolormesh(mesh_dim1, mesh_dim2, mod, cmap=cmap, vmin=color_min, vmax=color_max,
                            label='test1')
    
    # handle.set_rasterized(True)

    add_grid(ax)
    plot_addticks_cbar(cbar_title='$kg.m^{-3}$')

    ax.set_aspect('equal', 'box'),
    ax.set_xlabel('Distance along profile (km)')
    ax.set_ylabel('Depth (km)')

    plt.box(on=bool(1))
    plt.title(title_string)

    return handle


def extract_slice_from_vts(vts_file, axis='z', index=None):

    # TODO move this to an I/O module?

    # Check if file exists
    if not os.path.exists(vts_file):
        raise FileNotFoundError(f"VTS file not found: {vts_file}")

    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(vts_file)
    reader.Update()

    # Check if reader encountered errors
    if reader.GetErrorCode() != 0:
        raise RuntimeError(f"VTK reader error code: {reader.GetErrorCode()}")
    
    grid = reader.GetOutput()

    # Check if grid is valid
    if grid is None:
        raise RuntimeError("Failed to read VTS file - grid is None")
    
    if grid.GetNumberOfPoints() == 0:
        raise RuntimeError("VTS file appears to be empty - no points found")

    # Only extract the data we need, not the full array
    data = vtk_to_numpy(grid.GetPointData().GetScalars())

    dims = [0, 0, 0]
    grid.GetDimensions(dims)
    dims = tuple(dims)
    
    # Only extract the data we need, not the full array
    # Determine slice parameters before reshaping
    if axis == 'z':
        slice_idx = dims[2] // 2 if index is None else index
        # Reshape only the slice we need
        data = data.reshape((dims[2], dims[1], dims[0]))
        slice_2d = data[slice_idx, :, :]
    elif axis == 'y':
        slice_idx = dims[1] // 2 if index is None else index
        data = data.reshape((dims[2], dims[1], dims[0]))
        slice_2d = data[:, slice_idx, :]
    elif axis == 'x':
        slice_idx = dims[0] // 2 if index is None else index
        data = data.reshape((dims[2], dims[1], dims[0]))
        slice_2d = data[:, :, slice_idx]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return slice_2d


def save_slice_as_image(slice_2d0, slice_2d, padding_start, padding_end, x_misfit, y_misfit, y_lims, max_x_misfit, accept_ratio, gpars_, image_file_name, index, cmap=cc.cm.CET_R4, clim=np.array((2600, 3290)), ind=None, dpi=150):

    diff = slice_2d - slice_2d0

    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(6, 2)

    ax1 = fig.add_subplot(gs[0:2, 0:2])  
    # ax1.imshow(slice_2d, cmap=cmap, origin='lower')  # TODO make it pcolormesh

    horiz_axis = np.squeeze(gpars_.x.reshape(gpars_.dim)[:,index,padding_start:padding_end])
    vert_axis = -np.squeeze(gpars_.z.reshape(gpars_.dim)[::-1,index,padding_start:padding_end])

    plot_model(ax1, horiz_axis, vert_axis, 
                    slice_2d[:, padding_start:padding_end], 
                    '', cmap=cmap, clim=clim)

    if ind is not None: 
        # plt.text(3, 3, str(ind))
        if len(x_misfit)>1:
            ax1.set_title('Model num. ' + str(x_misfit[-1]))

    ax1_0 = fig.add_subplot(gs[2:4, 0:2])  

    plot_model(ax1_0, horiz_axis, vert_axis, 
                diff[:, padding_start:padding_end], 
                'Difference with prior', cmap='seismic', clim=np.array((-100, +100)))

    # Second plot on the third row
    ax2 = fig.add_subplot(gs[4, 0:2])    # only row 2

    ax2.plot(x_misfit, y_misfit)
    if len(x_misfit)>1:
        ax2.plot(x_misfit[-1], y_misfit[-1], 'o')

    ax2.set_xlim([0, max_x_misfit])
    ax2.set_ylim([y_lims[0], y_lims[1]])
    ax2.set_xlabel('Model index')
    ax2.set_title('Evolution of the data misfit')
    ax2.set_ylabel('Data misfit [mGal]')
    ax2.grid(True) 

    # Third plot on the fourth row
    ax3 = fig.add_subplot(gs[5, 0:5])    # only row 2

    ax3.plot(x_misfit, accept_ratio)
    if len(x_misfit)>1:
        ax3.plot(x_misfit[-1], accept_ratio[-1], 'o')

    ax3.set_xlim([0, max_x_misfit])
    ax3.set_ylim([0., 1.])
    ax3.set_xlabel('Model index')
    ax3.set_title('Evolution of the acceptance rate')
    ax3.set_ylabel('Uniteless')
    ax3.grid(True) 

    plt.tight_layout()
    
    # plt.axis('off')
    plt.savefig(image_file_name, dpi=dpi, bbox_inches=None)
    plt.close(fig)


def natural_sort_key(s):
    """ Function for natural sorting (e.g. 'file2' < 'file10'). """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

   
def add_grid(ax):
    """
    Add grid to existing plot axes.
    """

    # Add grid.
    ax.grid()
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='0.2', linestyle='--', alpha=0.1)
    ax.grid(visible=True, which='major', color='0.6', linestyle='--', alpha=1)


def set_plotprops():
    """
    Set default plot properties to use for all plots in the script.
    """

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 13})


def plot_addticks_cbar(cbar_title, cbar_ticks=None):
    """
    Add colorbar with specified title and ticks.

    :param cbar_title: title for the colorbar.
    :param cbar_ticks: location of the ticks on the colorbar.
    :return: colobar handle.
    """

    if cbar_ticks is None:

        cbar = plt.colorbar(shrink=0.75, orientation='vertical')
        cbar.set_label(cbar_title, labelpad=-20, y=+1.13, rotation=0, fontfamily='serif')
        # cbar.set_label(cbar_title, labelpad=-20, x=1.15, y=-0.02, rotation=0)
        # cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.125, rotation=0)

    else: 

        cbar = plt.colorbar(shrink=0.5, ticks=cbar_ticks, orientation='horizontal')
        cbar.set_label(cbar_title, labelpad=-20, y=+1.13, rotation=0, fontfamily='serif')
        # cbar.set_label(cbar_title, labelpad=-20, x=1.15, y=-0.02, rotation=0)
        # cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.125, rotation=0)

        ## Changing the font of ticks.
        # for i in cbar.ax.yaxis.get_title():
        #     i.set_family("Comic Sans MS")

    return cbar


def plot_data(geophy_data, geophy_data_diff):
    """   
    :param geophy_data_diff: 1D array, The forward geophy data of the due to the anomaly assessed using nullspace shuttle
    """ 

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(10, 10), constrained_layout=True)

    ax = fig.add_subplot(2, 2, 1)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data_diff)  # edgecolors='black')
    plt.scatter(geophy_data.x_data[np.abs(geophy_data_diff) > 1.5] / 1e3,
                geophy_data.y_data[np.abs(geophy_data_diff) > 1.5] / 1e3, 10,
                marker='.',
                color='k', linewidth=1)
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(a) Forward data of the perturbation')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(2, 2, 2)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_field)  # edgecolors='black')
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(b) Field data')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(2, 2, 3)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_calc)  # edgecolors='black')

    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(c) Calculated data')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    ax = fig.add_subplot(2, 2, 4)
    plt.scatter(geophy_data.x_data / 1e3,
                geophy_data.y_data / 1e3, 50, c=geophy_data.data_field - geophy_data.data_calc)  # edgecolors='black')
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    plt.title('(d) Residuals')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    plot_addticks_cbar(cbar_title='GeophyData (SI)')

    plt.show()

    return fig


def plot_navigation_xsection(mpars, ppars, ind_scatter):
    """
    Plots four 2D sections of the gridded model, along a specific slice, with a scatter plot on top for the last one.

    :param ind_scatter: tuple of length 2 with numpy.ndarray of indices along selected profile to use for scatter plot.
    :param mpars: ModelParameters object containing parameters for the model.
    :param ppars: PlotParameters object containing parameters for the plot.
    :return: None.
    """

    plot_models = ppars.plot_models
    colorschemes = ppars.colorschemes
    clims = ppars.clims
    cbar_ticks = ppars.cbar_ticks
    cbar_titles = ppars.cbar_titles
    plot_titles = ppars.plot_titles
    slice_x = ppars.slice_x

    # Get the coordinates for plotting.
    z_plot = mpars.z.reshape(mpars.dim)[:, ppars.slice_x, :]
    x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, :]

    n_subplots = 4
    n_row_subplots = 4
    n_columns_subplot = 1

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(13, 7))

    for i in range(0, n_subplots):

        ax = fig.add_subplot(n_row_subplots, n_columns_subplot, int(i + 1))
        plot_model(ax, mesh_dim1=x_plot, mesh_dim2=z_plot, mod=plot_models[i],
                   slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clims[i])
        plot_addticks_cbar(cbar_titles[i], cbar_ticks[i])
        
        plt.xlim((ppars.xlims[0], ppars.xlims[1]))

        ax.invert_yaxis()

    fig.tight_layout()
    plt.show()

    return fig


def prepare_plots(dim, mvars, m_diff, ppars, xlims, ylims):
    """
    Define plot parameters: models to plot, titles, limits etc.
    """

    print('\n Plot parameters are hardcoded in function', prepare_plots.__name__, "in file",  os.path.basename(__file__))

    # Models to plot in the 2x2 subplot.
    # First subplot.
    m1 = mvars.m_nullspace_orig.reshape(dim)
    # Second subplot.
    m2 = mvars.m_curr.reshape(dim)
    # Third subplot.
    m3 = mvars.m_geol_orig.reshape(dim)
    # Fourth subplot.
    m4 = m_diff.reshape(dim)

    ppars.plot_models = (m1, m2, m3, m4)

    # Color limits for the subplots.
    # For Pyrenees field case.
    # ppars.clims = (np.array([m1.min(), m1.max()]),  # In example shown in paper: m3 is the starting model.
    #                np.array([m1.min(), m1.max()]),
    #                np.array([m1.min(), m1.max()]),
    #                np.array([-200, 200]))
    # For homogenous model example.
    ppars.clims = (np.array([-300, 300]),
                   np.array([-300, 300]),
                   np.array([-300, 300]),
                   np.array([-300, 300]))

    # Colormaps for each subplot.
    ppars.colorschemes = (cc.cm.CET_R4,
                          cc.cm.CET_R4,
                          cc.cm.CET_R4,
                          'seismic')

    # Titles for each subplot.
    ppars.plot_titles = ('(a) Start of nullspace navigation: inverted model',
                         '(b) End of null space navigation',
                         '(c) Reference model for comparison',
                         '(d) Difference: End - Start')

    # Titles for each colorbar attached to the subplots.
    ppars.cbar_titles = ('$kg.m^{-3}$',
                         '$kg.m^{-3}$',
                         '$kg.m^{-3}$',
                         '$kg.m^{-3}$')

    # Ticks for each colorbar.
    # For homogenous model example.
    ppars.cbar_ticks = ([-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200])


    ppars.xlims = xlims
    ppars.ylims = ylims

    return ppars


# def save_plot(fig=None, filename='myplot', ext='.png', dpi=300, save=False):
#     """
#     Save the current figure to file or the figure provided in argument.

#     :param: filename (str): The name of the output file.
#     :param: dpi (int): Dots per inch (resolution) of the saved image (default: 300).
#     :param: format (str): The format of the output file (default: 'png').
#     :param: save (bool): Flag to indicate whether to save the plot (default: True).

#     Returns: None
#     """

#     filename = filename + ext

#     if save:

#         # Check that the extension provided is OK.
#         _, ext = os.path.splitext(filename)
#         ext_lower = ext.lower()

#         valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
#         if ext_lower not in valid_extensions:
#             raise ValueError("Invalid file extension. Supported extensions are: " + ", ".join(valid_extensions))

#         if fig is None:
#             # Get the current figure.
#             fig = plt.gcf()

#         # Do the saving;
#         fig.savefig(filename, dpi=dpi, format=ext_lower[1:])
