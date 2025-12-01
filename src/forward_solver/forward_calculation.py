"""
Forward Modeling Utilities.

This module provides functions and classes for computing forward data, 
residuals, and misfit gradients in geophysical.

Main Features
--------------
- **GeophyData**: Container for observed, calculated, and background
  data with coordinate and unit conversion options.
- **calc_fwd**: Computes the matrix x vector product of forward response.
- **calc_geophy_data**: handles forward data calculation.
- **calc_resid_vect**, **calc_rms**, **calc_data_rms**: Residual and
  RMS misfit computation utilities.
- **calc_grad_misfit**: Computes the gradient of the data misfit,
  including inverse weighting and optional wavelet reconstruction.

Notes
-----
- Differential modeling with `base_model` is not yet implemented.

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

import numpy as np
from numba import jit, njit
import math
from typing import Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.forward_solver.tomofast_sensit import Haar3D, iHaar3D, TomofastxSensit


@dataclass(slots=True)
class GeophyData:
    """
    Container for geophysical data, coordinates, and related background fields.

    Stores measured (data_field), calculated (data_calc), and background data,
    along with their spatial coordinates (x_data, y_data, z_data). Supports
    unit conversion and Bouguer anomaly correction.

    Parameters
    ----------
    unit_conv : bool
        Apply unit conversion (e.g., to mGal).
    bouguer_anomaly : bool, optional
        If True, subtract background for Bouguer anomaly. Default is True.
    data_calc : np.ndarray, optional
        Forward-calculated data (1D array).
    data_field : np.ndarray, optional
        Measured data (1D array), should match coordinate length.
    background : np.ndarray, optional
        Background field for anomaly calculation.
    x_data, y_data, z_data : np.ndarray, optional
        Coordinates of measurement points. All must be provided together.

    Notes
    -----
    - Warns if all values in data_field are identical.
    """
    unit_conv: bool
    bouguer_anomaly: bool = False
    data_calc: Optional[np.ndarray] = None
    data_field: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    x_data: Optional[np.ndarray] = None
    y_data: Optional[np.ndarray] = None
    z_data: Optional[np.ndarray] = None

    def __post_init__(self):
        # --- Coordinate completeness and consistency ---
        coords = [self.x_data, self.y_data, self.z_data]
        provided = [c is not None for c in coords]

        if any(provided) and not all(provided):
            raise ValueError("If one of x_data, y_data, or z_data is provided, all three must be set.")

        if all(provided):
            n_x_data, n_y_data, n_z_data = map(len, coords)
            if not (n_x_data == n_y_data == n_z_data):
                raise ValueError(f"Inconsistent coordinate lengths: "
                                 f"x_data={n_x_data}, y_data={n_y_data}, z_data={n_z_data}.")

        # --- Check consistency between coordinates and data_field ---
        if self.data_field is not None and all(provided):
            n_data = len(self.data_field)
            if len(self.x_data) != n_data:
                raise ValueError(f"Coordinate length ({len(self.x_data)}) does not match "
                                 f"data_field length ({n_data}).")

        # --- Warn if data_field has all identical values ---
        if self.data_field is not None:
            if np.allclose(self.data_field, self.data_field[0]):
                warnings.warn("All values in `data_field` are identical. "
                              "This may indicate missing or improperly loaded data - or some specific test.",
                                UserWarning, stacklevel=2)


def calc_fwd(sensitivity: TomofastxSensit, 
            model: np.ndarray, 
            unit_conv: bool, 
            indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculation of the forward data for model using sensitivity: d_calc = sensitivity * model.
    """
    # Convert to 1D view (fastest option)
    model_1d = model.ravel() if model.ndim != 1 else model

    use_csr_matrix = sensitivity.use_csr_matrix
    
    if use_csr_matrix:
        # Check for wavelet compression once
        if hasattr(sensitivity, 'compression_type') and sensitivity.compression_type == 1:
            # Wavelet case - must copy for reshaping
            model_work = model_1d.copy()
            if sensitivity.inv_weights is None: 
                model_work /= sensitivity.weight
            else: 
                model_work *= sensitivity.inv_weights
            
            # Wavelet transform
            shape = (sensitivity.nz, sensitivity.ny, sensitivity.nx)
            Haar3D(model_work.reshape(shape), *shape)
            # model_final = model_work.ravel()
        else:
            raise Exception("There should be wavelet transform!")
            # # No wavelet - can work with views
            # if indices is not None:
            #     model_final = model_1d / sensitivity.weight[indices]
            # else:
            #     model_final = model_1d / sensitivity.weight
        
        # Matrix multiplication
        # if indices is not None:
        #     data = sensitivity.matrix[:, indices] @ model_final[indices]

        data = sensitivity.matrix @ model_work.ravel()

    else:
        # Dense matrix case
        if indices is not None:
            data = sensitivity[:, indices] @ model_1d
        else:
            data = sensitivity @ model_1d

    # Unit conversion (1e2 for mGal)
    return data * 1e2 if unit_conv else data


def calc_geophy_data(geophy_data, sensitivity, dens_model, base_model=None):

    """
    Calculate the forward geophysical data.
    if base_model is provided, only the response of the difference with dens_model is calculated.

    :param geophy_data: geophy data dataclass.
    :param sensitivity: sensentivity matrix
    :param dens_model: density or density contrast model
    :param bouguer_anomaly: boolean controlling whether Bouguer anomaly is calculated
    :param use_csr_matrix:
    :param base_model:
    :return: None.

    Note: 
        Future work: consider the case where only a subset of the model (ie reduced model) is considered.
                     It could be that using indices array is not efficient and can cause memory issues with big matrices, see below:
                     https://stackoverflow.com/questions/39500649/sparse-matrix-slicing-using-list-of-int
    """

    unit_conv = geophy_data.unit_conv
    
    # Pre-convert background to avoid repeated array creation
    background_array = geophy_data.background if geophy_data.bouguer_anomaly else None

    # Flatten the 3D model to a 1D array.
    if dens_model.ndim != 1:
        dens_model = dens_model.flatten()

    if base_model is not None:
        raise Exception("base_model for diff calc not supported at the moment!")
        # # print('Calculate the data for the difference between the two models provided.')

        # # Flatten base model if needed
        # if base_model.ndim != 1:
        #     base_model = base_model.ravel()

        # # Find differences - only compute where they differ
        # diff_mask = base_model != dens_model
        # indices = np.where(diff_mask)[0]

        # # Only proceed if there are differences
        # if len(indices) > 0:
        #     diff_values = dens_model - base_model

        #     # Calculate forward data
        #     data_calc = calc_fwd(sensitivity, diff_values, unit_conv, indices)
            
        #     # Apply background correction if needed
        #     if geophy_data.bouguer_anomaly:
        #         data_calc = data_calc - background_array
        # else:
        #     # No differences, return zeros or background
        #     data_calc = -background_array if geophy_data.bouguer_anomaly else np.zeros(len(geophy_data.background))
            
    else:
        # No base model - calculate full forward
        if geophy_data.bouguer_anomaly:
            if background_array is None: 
                data_calc = calc_fwd(sensitivity, dens_model, unit_conv)
            else: 
                data_calc = calc_fwd(sensitivity, dens_model, unit_conv) - background_array
        else:
            # Note: sensitivity_reduced is not defined in original - this needs to be fixed
            # Assuming it should be 'sensitivity' for now
            data_calc = calc_fwd(sensitivity, dens_model, unit_conv)
    
        # Store result (no need for ravel since data_calc is already 1D)
        geophy_data.data_calc = data_calc
    
  
@njit
def calc_resid_vect(data_field, data_calc):
    """
    Calculate the between two vectors. Here, speficially the field data and calculated data. 
    :return residuals_vect: np.ndarray, the difference between field data and forward data.
    """

    return data_field - data_calc


@jit(nopython=True, cache=True)
def calc_rms(residuals_vector):
    """ 
    Compute the root mean square (RMS) of a residual vector.
    note: This function is the same as calc_rms in inversion_metrics module.
    """
    sum_sq = 0.0
    n = len(residuals_vector)
    for i in range(n):
        sum_sq += residuals_vector[i] * residuals_vector[i]
    return math.sqrt(sum_sq / n)


def calc_data_rms(geophy_data):
    """
    Calculate the root mean square difference (misfit) between calc_geophy_data and data_field
    and the calculated gravity data (data_calc), when geophy_data is a GravData class. Otherwise, when it is
    an array, it calculates the RMS value of geophy_data.

    :param geophy_data: GeophyData class containing the gravity data
    :return float, the residuals vector and data_misfit
    """

    if isinstance(geophy_data, np.ndarray):
        residuals_vect = geophy_data
    else:
        residuals_vect = calc_resid_vect(geophy_data.data_field, geophy_data.data_calc)

    return residuals_vect, calc_rms(residuals_vect)


def calc_grad_misfit(sensitivity, geophy_data):
    """
    Calculate the gradient of the data misfit function.
    Uses cached transpose for better performance.
    """
    # Calculate residuals vector.
    residuals_vector = calc_resid_vect(geophy_data.data_field, geophy_data.data_calc)
    
    if sensitivity.use_csr_matrix:
        # Use cached transpose property
        grad_misfit = -sensitivity.matrix_T.dot(residuals_vector)
        
        if sensitivity.compression_type == 1:
            # Apply inverse wavelet transform
            n1, n2, n3 = sensitivity.nz, sensitivity.ny, sensitivity.nx
            grad_misfit = grad_misfit.reshape(n1, n2, n3)
            iHaar3D(grad_misfit, n1, n2, n3)
            grad_misfit = grad_misfit.flatten()
        
        # Apply inverse depth weighting
        if sensitivity.inv_weights is not None:
            grad_misfit *= sensitivity.inv_weights
        else:
            grad_misfit /= sensitivity.weight
    else:
        grad_misfit = -np.matmul(sensitivity.T, residuals_vector)
    
    # Normalization
    grad_misfit /= len(residuals_vector)
    
    return grad_misfit
