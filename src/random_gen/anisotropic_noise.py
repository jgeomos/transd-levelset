"""
FFT-based correlated noise generation for 3D scalar field perturbations.

Provides tools for generating anisotropic 3D noise fields with controllable
correlation lengths and orientations using FFT methods. Used to create spatially correlated
perturbations for geometric model updates during inversion.

Main Components
---------------
FFTNoise : class
    Handles FFT processing and phase information for 3D noise fields
generate_noise_pert : function
    Generate correlated noise with specified correlation lengths and amplitude

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

import numpy as np
from numba import njit
from scipy.fft import ifftn, fftn


class FFTNoise():
    """
    Handles FFT-based processing of a 3D noise field and its phase information.

    Attributes:
        noise_ (np.ndarray): Input noise array (float32).
        N1_, N2_, N3_ (int): Dimensions of the input array.
        noise_fft_ (np.ndarray): FFT of the noise field.
        phase_info_ (np.ndarray): Normalized FFT phase (unit complex).
        u1_, u2_, u3_ (np.ndarray): Frequency grids for each axis.

    Args:
        noise_ (np.ndarray): Input noise of shape (N1_, N2_, N3_).
        N1_, N2_, N3_ (int): Dimensions of the noise array.

    Notes:
        - Adds small epsilon to avoid division by zero in phase normalization.
        - FFT is the main computational cost.
        - No shape validation yet — assumes noise_.shape matches N1_, N2_, N3_. TODO for future versions.
    """
 
    # TODO Future improvement: Add to FFTNoise class and test:
    # @property
    # def phase_info_(self):
    #     if self._phase_info_cache is None:
    #         self._phase_info_cache = self.noise_fft_ / (np.abs(self.noise_fft_) + 1e-10)
    #     return self._phase_info_cache

    noise_: np.array = None

    N1_: int = None
    N2_: int = None
    N3_: int = None

    u1_: np.array = None
    u2_: np.array = None
    u3_: np.array = None

    noise_fft_: np.array = None
    phase_info_: np.array = None

    def __init__(self, noise_, N1_, N2_, N3_):
        self.noise_ = noise_.astype(np.float32, copy=False)  
        self.N1_ = N1_
        self.N2_ = N2_
        self.N3_ = N3_

        self.noise_fft_ = fftn(self.noise_)  # This is the compute bottleneck. 50% of cost. Use GPU or 
        self.phase_info_ = self.noise_fft_ / (np.abs(self.noise_fft_) + 1e-10)  # + 1e-10 to avoid division by zero.

        # Define the frequency grid with quadrant shift; 
        # Shift the zero frequency component to the first index of the array.
        self.u1_ = np.fft.fftfreq(N1_).reshape(-1, 1, 1).astype(np.float32, copy=False)  
        self.u2_ = np.fft.fftfreq(N2_).reshape(1, -1, 1).astype(np.float32, copy=False)   
        self.u3_ = np.fft.fftfreq(N3_).reshape(1, 1, -1).astype(np.float32, copy=False)   


def normalize_to_symmetric_range(field, non_zero_indices, target_range=0.75):
    """
    Normalize field values to symmetric range [-target_range, +target_range].
    
    This function:
    1. Makes the field zero-mean
    2. Scales values to fit within [-target_range, +target_range]
    
    Args:
        field: Array to normalize (modified in-place)
        non_zero_indices: Indices of non-zero values to normalize
        target_range: Half-width of target range (default 0.75 for [-0.75, +0.75])
        
    Note:
        Modifies the input field array in-place.
        
    Warning:
        This normalization may be incompatible with subsequent amplitude scaling
        if amplitude_pert_min and amplitude_pert_max are also applied.
    """
    if len(non_zero_indices[0]) == 0:  # Handle case where non_zero_indices is empty
        return
    
    # Extract non-zero values for processing
    non_zero_values = field[non_zero_indices]
    
    # Step 1: Make zero-mean
    mean_val = np.mean(non_zero_values)
    field[non_zero_indices] -= mean_val
    
    # Recalculate after mean subtraction
    non_zero_values = field[non_zero_indices]
    min_val = np.min(non_zero_values)
    max_val = np.max(non_zero_values)
    
    # Step 2: Normalize to [-target_range, +target_range]
    if max_val != min_val:  # Avoid division by zero
        # Scale to [0, 1] then shift to [-0.5, +0.5] then scale by target_range
        normalized = 2 * (non_zero_values - min_val) / (max_val - min_val) - 1
        field[non_zero_indices] = normalized * target_range
    else:
        # All values are the same, set to zero
        field[non_zero_indices] = 0.0


@njit(cache=True, fastmath=True)
def normalise_by_std(field_):
    """Calculate (field_ - mean(field_))/std(field_)"""
    field_32 = field_.astype(np.float32)
    return ((field_32 - np.mean(field_32)) / np.std(field_32))


def compute_squared_radius(u1, u2, u3, data_type='float32'):
    """
    Compute the squared radius (u1² + u2² + u3²) from 3D frequency grids.

    Args:
        u1, u2, u3 (np.ndarray): Frequency components along each axis, broadcastable to the same shape.
        data_type (str): Output data type; defaults to 'float32'. If not 'float32', original dtype is kept.

    Returns:
        np.ndarray: Squared radius array of the same shape as inputs, optionally cast to float32.
    """
    squared_radius = u1 * u1
    np.add(squared_radius, u2 * u2, out=squared_radius)
    np.add(squared_radius, u3 * u3, out=squared_radius)
    if data_type == 'float32':
        return squared_radius.astype(np.float32)
    else: 
        return squared_radius


@njit(cache=True, fastmath=True)
def compute_filter(squared_radius, power_exponent, det):
    """
    Compute the spectral filter for correlated 1/f noise in the frequency domain.

    Applies a power-law decay to the squared frequency radius and scales by
    the determinant of the correlation matrix.

    Args:
        squared_radius (np.ndarray): Squared frequency radius (u1² + u2² + u3²).
        power_exponent (float): Exponent applied to the frequency radius (typically 0.5 * factor_spectrum).
        det (float): Determinant of the correlation matrix, used for normalization.

    Returns:
        np.ndarray: Filter array in frequency space (float32).
    """
    return (1.0 / ((squared_radius ** power_exponent + 1e-4) * det)).astype(np.float32)


@njit(fastmath=True)
def calc_det_3x3(matrix):
    """
    Compute the determinant of a 3x3 matrix using the cofactor expansion.
    Args:
        matrix (np.ndarray): A 3x3 matrix (assumed shape: (3, 3)).
    Returns:
        float: Determinant of the input matrix.
    """
    return (matrix[0,0] * (matrix[1,1] * matrix[2,2] - matrix[1,2] * matrix[2,1]) - 
            matrix[0,1] * (matrix[1,0] * matrix[2,2] - matrix[1,2] * matrix[2,0]) + 
            matrix[0,2] * (matrix[1,0] * matrix[2,1] - matrix[1,1] * matrix[2,0]))


@njit(cache=True, fastmath=True)
def transform_coordinates(transform_matrix, u1, u2, u3):
    """
    Apply 3x3 transformation matrix to coordinate arrays.
    Replaces the manual element extraction and transformation.
    """
    # Extract transformation matrix elements
    t00, t01, t02 = transform_matrix[0, 0], transform_matrix[0, 1], transform_matrix[0, 2]
    t10, t11, t12 = transform_matrix[1, 0], transform_matrix[1, 1], transform_matrix[1, 2]
    t20, t21, t22 = transform_matrix[2, 0], transform_matrix[2, 1], transform_matrix[2, 2]
    
    # Apply transformation
    u1_new = t00 * u1 + t01 * u2 + t02 * u3
    u2_new = t10 * u1 + t11 * u2 + t12 * u3
    u3_new = t20 * u1 + t21 * u2 + t22 * u3
    
    return u1_new, u2_new, u3_new


def create_anisotropic_noise_field(dimensions, factor_spectrum, rng,
                                  correlation_length=np.array((1.,1.,1.)), 
                                  corr_zx=0., corr_zy=0., corr_xy=0., 
                                  rotation_angle=(0.,0.,0.), cell_sizes=None, 
                                  field_to_use=None, rotation_matrix=None, 
                                  fft_class=None, fftw_instance=None):
    """
    Generates a 3D anisotropic 1/f^gamma noise field with correlations (x-y, x-z, and y-z) 
    and rotations.

    Original 2D isotropic case ('noiseonf') from P. Kovesi, Univ. of Western Australia.
    For original homogenous 2D code: Copyright (c) 1996-2014 Peter Kovesi, Centre for Exploration Targeting
    The University of Western Australia, peter.kovesi at uwa edu au

    Ported to Python and extended to 3D and by Jeremie Giraud, March 2022.
    Extended to correlated, anisotropic case with rotations by Jeremie Giraud, March 2025.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    The Software is provided "as is", without warranty of any kind.

    December  1996
    March     2009 Arbitrary image size
    September 2011 Code tidy up
    April     2014 Fixed to work with odd dimensioned images
    March     2022 Ported to Python and extended to 3D
    March     2025 Extended to anisotropic, correlated noise
    Mary      2025 Rotation, normalisation.
    """

    if len(dimensions) != 3:
        print(dimensions)
        raise ValueError("dimensions should correspond to a 3D model!")

    N1, N2, N3 = dimensions[0], dimensions[1], dimensions[2]

    # Generate white noise
    if field_to_use is None and fft_class is None:
        noise = rng.normal(size=(N1, N2, N3))
    else: 
        noise = field_to_use

    if fft_class is None:  # TODO cache a FFT class for performance.
        fft_class = FFTNoise(noise, N1, N2, N3)
        u1, u2, u3, phase_info = fft_class.u1_, fft_class.u2_, fft_class.u3_, fft_class.phase_info_
    else: 
        u1, u2, u3, phase_info = fft_class.u1_, fft_class.u2_, fft_class.u3_, fft_class.phase_info_

    # Apply scaling: account for difference in cell size in x, y, and z dir
    if cell_sizes is not None: 
        u1 /= cell_sizes[0]
        u2 /= cell_sizes[1]
        u3 /= cell_sizes[2]

    # Define the correlation matrix
    # The Diagonal elements are an application of anisotropic scaling
    correlation_matrix = np.array([
        [correlation_length[0], corr_zx, corr_zy],  # dim0 correlation with dim1, 2
        [corr_zx, correlation_length[1], corr_xy],  # dim1 correlation with dim0, 2
        [corr_zy, corr_xy, correlation_length[2]]   # dim2 correlation with dim0, 1
    ])
    
    # Compute the determinant
    if not (corr_zx==0 and corr_zy==0 and corr_xy==0):
        det = calc_det_3x3(correlation_matrix)
    else: 
        det = correlation_matrix.diagonal().prod()

    # inv_corr =  np.linalg.inv(correlation_matrix)
    
    # Get the rotation matrix
    if rotation_matrix is None: 
        """ Deactivated in this version """
        rotation_matrix = np.eye(3)

    # Calculate the matrix applying both the rotation and correlation
    """ Deactivated in this version """
    transform_coord = np.eye(3)

    # Split into components
    u1, u2, u3 = transform_coordinates(transform_coord, u1, u2, u3)

    # Apply correlated power spectrum
    # Calcualte frequency radius (squared). Faster than squared_radius = u1*u1 + u2*u2 + u3*u3
    squared_radius = compute_squared_radius(u1, u2, u3, data_type='float32')
    squared_radius = squared_radius.astype(np.float32, copy=False)

    # Construct correlated filter in frequency space
    power_exponent = factor_spectrum * 0.5
    filter_ = compute_filter(squared_radius, power_exponent, det)

    # Apply filter in frequency domain
    # filtered_fft = filter_ * phase_info
    if fftw_instance is None: 
        field = np.real(ifftn(filter_ * phase_info, overwrite_x=True))
    else:
        field = fftw_instance.fast_ifftn(filter_, phase_info)
    
    field = normalise_by_std(field)

    return field