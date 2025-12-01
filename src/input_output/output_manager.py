"""
A module containing functions and classes for looging the execution of runs and saving outputs.
"""

import logging
import numpy as np
import vtk
import glob
import os
import json


class ConditionalFormatter(logging.Formatter):
    """
    Custom log formatter that omits the log level for INFO messages,
    but includes it for WARNING, ERROR, and higher levels.
    Formats timestamps without milliseconds.
    """
    def __init__(self):
        # Initialize with datefmt to remove milliseconds
        super().__init__(datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        if record.levelname == 'INFO':
            # Format without levelname for INFO messages
            self._style._fmt = '%(asctime)s - %(message)s'
        else:
            # Format with levelname for WARNING, ERROR, etc.
            self._style._fmt = '%(asctime)s - %(levelname)s - %(message)s'
        return super().format(record)


class LogRun:
    """
    Logging utility class for managing console and file logging during a script run.

    Use return_non=True to not do logging.

    Args:
        log_name (str): Filename for the log output.
        logger_name (str): Internal name of the logger.
        to_console (bool): Whether to also print log messages to the console.

    Methods:
        info(message): Logs an info-level message.
        warning(message): Logs a warning-level message.
        error(message): Logs an error-level message.
        close(): Closes all log handlers and finalizes logging.
    """

    def __new__(cls, log_name='simulation_log.txt', logger_name='logrun', to_console=True, verbose=True, return_none=False):
        """
        This conditional disabling of logging. Create and return a new LogRun instance. If return_none is True, skip object
        creation and return None instead. 
        """
        print('Logging: deactivated. ')
        if return_none:
            print('Logging: deactivated.\n')
            return None  # prevents instance creation entirely
        return super().__new__(cls)  # create a normal instance

    def __init__(self, log_name='simulation_log.txt', logger_name='logrun', to_console=True, verbose=True, return_none=False):
        # If return_none=True, __init__ won’t run because __new__ returned None
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers (important for notebooks)
        self.verbose = verbose

        file_handler = logging.FileHandler(log_name, mode='w')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        if to_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_format = logging.Formatter('%(message)s')
            stream_handler.setFormatter(stream_format)
            self.logger.addHandler(stream_handler)

        if self.verbose:
            self.info('\n (Logging results: can take slightly longer and create big log files.)')
            self.info('\n--------- Started main script ---------')

    def info(self, message):
        if self.verbose:
            self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def close(self):
        self.info('Completed.')
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def log_exception(self, e, context=""):
        error_msg = f"{context}: {type(e).__name__}: {str(e)}"
        self.error(error_msg)
        raise e
    

def save_random_field_perturbation_outputs(i, mvars, pert, mask_force_pert, gpars, spars, log_run):
    """
    For use when perturbations using random field fail, prior to exiting main script.

    Save model and perturbation-related arrays to VTK files with consistent naming.

    Parameters:
        i (int): Iteration index.
        mvars (object): Contains current model (`m_curr`) and auxiliary model (`mod_aux`).
        pert (np.ndarray): Random field perturbation array.
        mask_force_pert (np.ndarray): Binary mask where perturbations were forced.
        gpars (object): Grid parameters for VTK export.
        spars (object): Contains output paths and save flags.
        log_run (logging.Logger): Logger to use for save logs.
    """
    base_path = spars.path_output
    save_flag = spars.save_plots

    save_model_to_vtk(
        mvars.m_curr, gpars,
        filename=f"{base_path}/pb_{spars.filename_model_save_rt}{i}",
        save=save_flag, log_run=log_run
    )
    save_model_to_vtk(
        mvars.mod_aux, gpars,
        filename=f"{base_path}/pb_{spars.filename_aux_save_rt}{i}",
        save=save_flag, log_run=log_run
    )
    save_model_to_vtk(
        mask_force_pert, gpars,
        filename=f"{base_path}/pb_mask{spars.filename_aux_save_rt}{i}",
        save=save_flag, log_run=log_run
    )
    save_model_to_vtk(
        pert, gpars,
        filename=f"{base_path}/pb_pert{spars.filename_aux_save_rt}{i}",
        save=save_flag, log_run=log_run
    )


def remove_vtk_files(folder_path, target_string, target_extension=".vtp", verbose=False, log_run=None):
    """
    Removes VTK-related files in a folder that match a target string and extension.

    Args:
        folder_path (str): Path to the folder containing the files.
        target_string (str): Substring to search for in filenames (use "*" to delete all matching the extension).
        target_extension (str): File extension to filter by (default ".vtp").
        verbose (bool): If True, prints/logs the number of deleted files.
        log_run (LogRun, optional): Optional LogRun instance for logging instead of printing.

    Returns:
        None
    """

    nfiles = 0

    if target_string == "*":
        # Find all files with the given extension
        files_to_delete = glob.glob(os.path.join(folder_path, f"*{target_extension}"))
        # Delete the files
        for file_path in files_to_delete:
            if os.path.isfile(file_path):
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
                nfiles += 1

    else: 
        # Loop through files and remove ones containing the target string.
        for filename in os.listdir(folder_path):
            if target_string in filename:
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    nfiles += 1
                    # print(f"Removed: {file_path}")

    if verbose:
        if log_run is not None: 
            log_run.info(f'Deleted {nfiles} files')
        else: 
            print('Deleted ' + str(nfiles) + ' files')
    
    return None


def save_data_to_vtk(geophy_dataclass, datatype_to_save='data_calc', filename='data_file', save=True, log_run=None):
    """
    Saves geophysical data from a GeophyData object to a VTK (.vtp) file for visualization with e.g. Paraview.
    Args:
        geophy_dataclass (GeophyData): The data container with spatial and scalar fields.
        datatype_to_save (str): Which data to save. Options:
            - 'data_field': Original data
            - 'data_calc': Calculated/simulated data
            - 'background': Background model
            - 'difference': Difference between 'data_field' and 'data_calc'
        filename (str): Output filename (without extension).
        save (bool): If False, skips saving but logs/prints that file was not saved.
        log_run (LogRun, optional): Optional logger for structured output.

    Returns:
        None
    """

    if save:
        x = geophy_dataclass.x_data
        y = geophy_dataclass.y_data
        z = geophy_dataclass.z_data

        if datatype_to_save == 'data_field':
            values = geophy_dataclass.data_field
        elif datatype_to_save == 'data_calc':
            values = geophy_dataclass.data_calc
        elif datatype_to_save == 'background':
            values = geophy_dataclass.background
        elif datatype_to_save == 'difference':
            values = geophy_dataclass.data_field - geophy_dataclass.data_calc
        else: 
            e = Exception("datatype_to_save can only be data_field, data_calc, or background")
            if log_run:
                log_run.log_exception(e, "datatype_to_save can only be data_field, data_calc, or background")
            else:
                raise e

        num_points = len(values)

        # Create a VTK points object.
        points = vtk.vtkPoints()
        for i in range(num_points):
            points.InsertNextPoint(x[i], y[i], z[i])

        # Create a PolyData object and set the points.
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)    

        # Add scalar data (optional)
        scalars_array = vtk.vtkFloatArray()
        scalars_array.SetName("GeophyData")  # Name appears in ParaView
        for s in values:
            scalars_array.InsertNextValue(s)

        poly_data.GetPointData().SetScalars(scalars_array)

        # Write to VTK file (.vtp format)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + ".vtp")
        writer.SetInputData(poly_data)
        writer.Write()
        del writer  # Explicit cleanup
        poly_data.ReleaseData()  # Release VTK data

        if log_run is not None:
            log_run.info(f"VTK data file saved as: {filename}.vtp")
        # else: 
        #     print("VTK data file saved as: " + filename + ".vtp")
    else:
        if log_run is not None:
            log_run.info(f"VTK data file saved as: {filename}.vtp")
        # else: 
        #     print("VTK file for data " + datatype_to_save + " NOT saved")

    return None


def save_acceptance_history(metrics: "MonitoringMetrics", run_params: "RunBaseParams", 
                            filename: str = "accepted_changes", save: bool = True, 
                            log_run: object = None):
    """
    Save acceptance tracking data into text summary files.

    Parameters
    ----------
    metrics : MonitoringMetrics
        Object containing acceptance tracking data.
    run_params : RunBaseParams
        Object containing perturbation type definitions.
    filename : str, optional
        Output filename without extension. Default is 'accepted_changes'.
    save : bool, optional
        If True, saves the files. Default is True.
    log_run : LogRun, optional
        Logger instance for status messages.
    
    Returns
    -------
    None
    """
    if save:
        summary = metrics.get_acceptance_summary(run_params)

        # Save full data in a tab-delimited text file
        data_file = f"{filename}.txt"
        with open(data_file, 'w') as f:
            f.write("Accepted Iterations\tAccepted Types\n")
            for it, typ in zip(metrics.it_accepted_model, metrics.it_accepted_type):
                f.write(f"{int(it)}\t{int(typ)}\n")

        if log_run:
            log_run.info(f"Acceptance history saved in: {data_file}")

        # Save human-readable summary
        summary_file = f"{filename}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ACCEPTED CHANGES SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total accepted: {summary['total_accepted']}\n\n")

            for name, info in summary['by_type'].items():
                if info['count'] > 0:
                    f.write(f"{name}: {info['count']} times\n")
                    f.write(f"  Iterations: {info['iterations']}\n\n")

        if log_run:
            log_run.info(f"Acceptance summary saved in: {summary_file}")

    else:
        if log_run:
            log_run.info("Acceptance history NOT saved")

    return None


def save_metrics_txt(metrics: "MonitoringMetrics", filename: str = "metrics_file", 
                           save: bool = True, log_run: object = None):

    """
    Save monitoring metrics into a text file.

    Parameters
    ----------
    metrics : MonitoringMetrics
        Object containing arrays/lists of monitoring values.
    filename : str, optional
        Output filename without extension. Default is 'metrics_file'.
    save : bool, optional
        If True, saves the txt file. Default is True.
    log_run : object, optional
        Logger for status messages.
    """

    if save:
        n_iter = len(metrics.data_misfit)
        it_nums = np.arange(1, n_iter + 1, dtype=int)

        column_names = [
            "Iteration number",
            "Data misfit",
            "Model misfit",
            "Petrophysical misfit",
            "Acceptance ratio",
            "Log-likelihood ratio",
            "Log-prior ratio",
            "Log-petrophysical ratio"
        ]

        values = [
            it_nums,
            metrics.data_misfit,
            metrics.model_misfit,
            metrics.petro_misfit,
            metrics.accept_ratio,
            metrics.log_likelihood_ratio,
            metrics.log_priorgeom_ratio,
            metrics.log_priorpetro_ratio
        ]

        # Stack columns into 2D array
        data = np.column_stack(values)
        
        # Tab-separated header
        header = '\t'.join(column_names)
        
        # Save file with tab delimiter
        tsv_filename = filename + ".txt"
        np.savetxt(tsv_filename, data, delimiter='\t', header=header,
                   comments='', fmt='%.6f')

        if log_run is not None:
            log_run.info(f"Monitoring metrics saved in: {tsv_filename}")
    else:
        if log_run is not None:
            log_run.info("Monitoring metrics NOT saved")
    
    return None


def save_model_to_vtk(voxel_data, grid_par_class, filename='voxet', save=True, log_run=None):
    """
    Create VTK structured grid from voxel data using the format used in the Nullspace script.
    
    voxel_data: Voxel model to save.
    grid_par_class: GridParameters dataclass.
    Needs changing dimension order if input follows Python: VTK expects Fortran-style (column-major) ordering for structured grids.
    """

    if voxel_data is None: 
        save = False
        if log_run is not None: 
            log_run.info(f"VTK file {filename} NOT saved")
            log_run.warning(f"Voxel_data is None!")
        else: 
            print(f"VTK file {filename} not saved: voxel_data is None!")
    else: 
        voxel_data = voxel_data.flatten()
    
    if not save:
        if log_run is not None: 
            log_run.info(f"VTK file {filename} NOT saved")
            log_run.warning(f"VTK file {filename}: save = False")
        # else: 
        #     print("VTK file " + filename + " not saved")
        return None
    
    voxel_data = voxel_data.flatten() if voxel_data.ndim > 1 else voxel_data
    
    # Change dimension order from row-major to column-major
    mesh_dims = np.array([grid_par_class.dim[2], grid_par_class.dim[1], grid_par_class.dim[0]], dtype=np.int32)
    
    # Create the structured grid object
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(mesh_dims[0], mesh_dims[1], mesh_dims[2])
    
    # Reshape coordinate arrays once
    x_grid = grid_par_class.x.reshape(mesh_dims)  # Here, swap x and y from Python to Fortran and vice-versa. TODO fix this. 
    y_grid = grid_par_class.y.reshape(mesh_dims)  # Here, swap x and y from Python to Fortran and vice-versa. TODO fix this. 
    z_grid = grid_par_class.z.reshape(mesh_dims)
    
    # OPTIMIZATION 1: Vectorized point creation
    # Flatten and stack coordinates for vectorized operations
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()[::-1]
    z_flat = z_grid.flatten()[::-1]
    
    # Stack coordinates into a single array (N x 3)
    coords = np.column_stack((x_flat, y_flat, z_flat)).astype(np.float32)
    
    # Create VTK points array directly from numpy array
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(coords))
    structured_grid.SetPoints(vtk_points)
    
    # OPTIMIZATION 2: Direct numpy to VTK array conversion.
    # Ensure voxel_data is contiguous and correct dtype.
    voxel_data_opt = np.ascontiguousarray(voxel_data.flatten(), dtype=np.float32)
    
    # Convert NumPy array to VTK array using numpy_support for better performance
    vtk_array = vtk.util.numpy_support.numpy_to_vtk(voxel_data_opt)
    vtk_array.SetName("PhysPropertyValue")
    
    # Assign data to structured grid.
    structured_grid.GetPointData().SetScalars(vtk_array)
    
    # OPTIMIZATION 3: Use binary format for faster I/O.
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename + ".vts")
    writer.SetInputData(structured_grid)
    writer.SetDataModeToBinary()  # Binary format is faster than ASCII
    writer.SetCompressorTypeToZLib()  # Optional: compress for smaller files
    writer.Write()

    if log_run is not None: 
        log_run.info(f"Saved VTK file {filename}")
        # else: 
        #     print("Saved VTK file " + filename)
    
    return None


def save_model_tomofast(orig_tomofast_file, model_values, destination_file="data/models/your_file_name.txt", log_run=None):
    """
    Reads a file written using Tomofast format and replaces the model values with the provided model. 
    """
    
    # m_curr should be the last model of your perturbations. 
    model_to_save = model_values.flatten()

    # Read the original input file to get grid information. 
    with open(orig_tomofast_file, "r") as f:
        lines = f.readlines()

    # Replace the 7th column (index 6, ie model values) in each data row
    header = lines[0]
    data_lines = lines[1:]
    modified_lines = []
    for i, line in enumerate(data_lines):
        parts = line.strip().split()  # or use `split(',')` if comma-separated
        if len(parts) < 7:
            if log_run is not None: 
                e = Exception(f"Line {i+2} has fewer than 7 columns: {line}")
                log_run.log_exception(e, f"Line {i+2} has fewer than 7 columns: {line}")
            else: 
                raise ValueError(f"Line {i+2} has fewer than 7 columns: {line}")
        parts[6] = str(model_to_save[i])  # Replace 7th column (index 6)
        modified_lines.append(" ".join(parts) + "\n")

    # Check if the destination directory exists
    destination_dir = os.path.dirname(destination_file)
    if not os.path.exists(destination_dir):
        raise FileNotFoundError(f"Destination directory '{destination_dir}' does not exist.")

    # Save to a new file
    with open(destination_file, "w") as f:
        f.write(header)
        f.writelines(modified_lines)


def write_tomofast_model_grid(line_data, output_folder="tomofast_grids_test", file_name='model_grid_test.txt'):
    """
    A function to write the model and grid for Tomofast-x.
    """

    filename = output_folder + "/" + file_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    num_cells = line_data.shape[0]
    print("num_cells = ", num_cells)

    with open(filename, "w") as file:
        # Write the header.
        file.write("%d\n" % num_cells)

        np.savetxt(file, line_data, fmt="%f %f %f %f %f %f %f %d %d %d")
    file.close()


def write_tomofast_data_grid(line_data, 
                        output_folder="tomofast_grids_test", 
                        file_name='data_grid_test.txt', 
                        fmt="%.6f", 
                        delimiter=" "):
    """
    Save a 4-column array (x, y, z, data) to a text file.
    Writes row count as first line.
    """
    
    destination_file = output_folder + "/" + file_name

    # Check if the destination directory exists
    destination_dir = os.path.dirname(destination_file)
    if not os.path.exists(destination_dir):
        raise FileNotFoundError(f"Destination directory '{destination_dir}' does not exist.")
    
    # Validation
    if not isinstance(line_data, np.ndarray):
        raise TypeError("Input 'data' must be a numpy array.")
    if line_data.ndim != 2 or line_data.shape[1] != 4:
        raise ValueError("Input 'data' must have shape (N, 4).")

    # Write file
    nrows = line_data.shape[0]
    with open(destination_file, "w") as f:
        f.write(f"{nrows}\n")
        np.savetxt(f, line_data, fmt=fmt, delimiter=delimiter)

    print(f"✅ Saved {nrows} rows to {destination_file}")


def save_metrics_summary(metrics, filepath, type_labels=None):
    """
    Save summary statistics to a text file.
    
    Parameters
    ----------
    metrics : InversionMetrics
        Metrics object with tracking data
    filepath : str
        Path to output file
    type_labels : list of str, optional
        Labels for perturbation types
    """
    if type_labels is None:
        type_labels = [f"Type {i}" for i in range(5)]
    
    with open(filepath, 'w') as f:
        # Header with summary statistics
        f.write("Inversion Metrics Summary\n")
        f.write(f"Reference misfit: {metrics.reference_misfit}\n")
        f.write(f"Last accepted misfit: {metrics.last_misfit_accepted}\n")
        f.write("\n")
        
        # Acceptance statistics
        n_total = len(metrics.data_misfit)
        n_accepted = len(metrics.it_accepted_model)
        acceptance_rate = n_accepted / n_total if n_total > 0 else 0
        
        f.write(f"Acceptance rate: {acceptance_rate:.3f}\n")
        f.write(f"Mean accept ratio: {np.mean(metrics.accept_ratio):.3f}\n")
        f.write("\n")
        
        # Breakdown by type
        f.write("Accepted changes by type:\n")
        for pert_type in range(5):
            count = metrics.count_by_type(pert_type)
            if count > 0:
                percentage = count / n_accepted * 100 if n_accepted > 0 else 0
                f.write(f"  {type_labels[pert_type]}: {count} ({percentage:.1f}%)\n")


def save_metrics_data(metrics, filepath):
    """
    Save detailed iteration data to CSV file.
    
    Parameters
    ----------
    metrics : InversionMetrics
        Metrics object with tracking data
    filepath : str
        Path to output CSV file
    """
    with open(filepath, 'w') as f:
        # Column headers
        f.write("iteration,data_misfit,model_misfit,petro_misfit,accept_ratio,"
                "log_likelihood_ratio,log_priorgeom_ratio,log_priorpetro_ratio,n_units_total\n")
        
        # Data rows
        n_total = len(metrics.data_misfit)
        for i in range(n_total):
            f.write(f"{i},{metrics.data_misfit[i]},{metrics.model_misfit[i]},"
                   f"{metrics.petro_misfit[i]},{metrics.accept_ratio[i]},"
                   f"{metrics.log_likelihood_ratio[i]},{metrics.log_priorgeom_ratio[i]},"
                   f"{metrics.log_priorpetro_ratio[i]},{metrics.n_units_total[i]}\n")