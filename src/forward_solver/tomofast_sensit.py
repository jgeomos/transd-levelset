"""
Module for handling Tomofast-x sensitivity matrices, the related wavelet transform, 
and depth weighting. 

Main components
---------------
- TomofastxSensit : Dataclass container for Tomofast-X sensitivity matrices and weights.
- Haar3D          : Forward 3D Haar wavelet transform.
- iHaar3D         : Inverse 3D Haar wavelet transform.

Authors:
    Vitaliy Ogarko
    Jérémie Giraud
    
License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"

from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import spmatrix
import struct
import os
from numba import jit


# Constants for Haar forward and inverse transforms.
SQRT2 = np.sqrt(2.0)
INV_SQRT2 = 1.0 / np.sqrt(2.0)
LOG2 = np.log(2.)


@dataclass
class TomofastxSensit:
    """
    Sensitivity matrix container for modelling using Tomofast-x format with wavelet compression.
    
    Attributes
    ----------
    nx, ny, nz : int
        Model dimensions (number of cells in each direction)
    compression_type : int
        Compression type (0=none, 1=wavelet/Haar, 2=other - currently unsupported)
    matrix : scipy.sparse
        Sensitivity matrix in CSR format, shape (n_data, nx*ny*nz)
    weight : scipy.sparse
        Depth weighting factors for each cell
    inv_weights : np.ndarray
        Cached 1/weight for efficiency (computed via calc_inv_weights)
    use_tomofast_sensit : bool
        Whether to use Tomofast sensitivity format (always True currently)
    _matrix_T_cached : scipy.sparse
        Cached transpose for gradient calculations (internal use)
    
    Methods
    -------
    calc_inv_weights()
        Pre-compute inverse weights
    matrix_T : property
        Get cached transpose (lazy-loaded)
    invalidate_cache()
        Clear transpose cache if matrix changes
    precompute_all()
        Pre-compute all cached values for optimization
    
    Notes
    -----
    The matrix is typically very sparse (typically >95% zeros). Caching the transpose
    roughly doubles memory but is expected to provide speedup in calculation of gradient of cost function.
    """

    # Dimensions of the model. 
    nx: int
    ny: int
    nz: int
    
    compression_type: int
    matrix: spmatrix
    weight: spmatrix 
    inv_weights: np.array = None
    use_tomofast_sensit: bool = field(default=True, init=False)
    use_csr_matrix: bool = field(default=True, init=False)
    _matrix_T_cached: object = field(default=None, init=False, repr=False)  # Hidden cache field
    
    def calc_inv_weights(self):
        """Pre-calculate inverse depth weight for use in forward calculation."""
        self.inv_weights = 1.0 / self.weight
    
    @property
    def matrix_T(self):
        """Lazy-load and cache the transposed matrix in optimal format."""
        if self._matrix_T_cached is None:
            # Create transpose and optimize for row operations
            self._matrix_T_cached = self.matrix.T.tocsr()
        return self._matrix_T_cached
    
    # def invalidate_cache(self):
    #     """Clear the cached transpose if matrix changes."""
    #     self._matrix_T_cached = None
    
    def precompute_all(self):
        """Pre-compute all expensive operations at once."""
        # Compute inverse weights for later use. 
        if self.inv_weights is None:
            self.calc_inv_weights()
        
        # Pre-cache the transpose for later use. 
        _ = self.matrix_T  # This triggers the lazy loading
        
        return self
    
    def __post_init__(self):
        if not self.use_csr_matrix:
            raise Exception("Dense matrix not supported at the moment!")
    

@jit(nopython=True, cache=True)
def Haar3D(s, n1, n2, n3):
    """
    Forward Haar wavelet transform.
    """
    
    dims = np.array([n1, n2, n3])
    
    for ic in range(3):
        L = dims[ic]
        n_scale = int(np.log2(L))
        
        for istep in range(1, n_scale + 1):
            step_incr = 1 << istep
            half_step = step_incr >> 1
            
            ngmin = half_step + 1
            ngmax = ngmin + ((L - ngmin) // step_incr) * step_incr
            ng = (ngmax - ngmin) // step_incr + 1
            
            # Combined operations in single loop
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    # All operations for axis 0
                    for j in range(n2):
                        for k in range(n3):
                            s[ig, j, k] -= s[il, j, k]
                            s[il, j, k] += s[ig, j, k] * 0.5
                            s[il, j, k] *= SQRT2
                            s[ig, j, k] *= INV_SQRT2
                elif ic == 1:
                    # All operations for axis 1
                    for j in range(n1):
                        for k in range(n3):
                            s[j, ig, k] -= s[j, il, k]
                            s[j, il, k] += s[j, ig, k] * 0.5
                            s[j, il, k] *= SQRT2
                            s[j, ig, k] *= INV_SQRT2
                else:
                    # All operations for axis 2
                    for j in range(n1):
                        for k in range(n2):
                            s[j, k, ig] -= s[j, k, il]
                            s[j, k, il] += s[j, k, ig] * 0.5
                            s[j, k, il] *= SQRT2
                            s[j, k, ig] *= INV_SQRT2
                
                il += step_incr
                ig += step_incr

#=========================================================================================
@jit(nopython=True, cache=True)
def iHaar3D(s, n1, n2, n3):
    """
    Inverse Haar transform.
    """
    
    for ic in range(3):
        if ic == 0:
            n_scale = int(np.log(float(n1)) / LOG2)
            L = n1
        elif ic == 1:
            n_scale = int(np.log(float(n2)) / LOG2)
            L = n2
        else:
            n_scale = int(np.log(float(n3)) / LOG2)
            L = n3

        for istep in range(n_scale, 0, -1):  # Reversed range for inverse
            step_incr = 2 ** istep
            ngmin = int(step_incr / 2) + 1
            ngmax = ngmin + int((L - ngmin) / step_incr) * step_incr
            ng = int((ngmax - ngmin) / step_incr) + 1

            # Combined operations in reverse order
            ig = ngmin - 1
            il = 0
            for i in range(ng):
                if ic == 0:
                    for j in range(n2):
                        for k in range(n3):
                            # Inverse order: Normalize, Update, Predict
                            s[il, j, k] *= INV_SQRT2  # Normalize
                            s[ig, j, k] *= SQRT2
                            s[il, j, k] -= s[ig, j, k] * 0.5  # Update
                            s[ig, j, k] += s[il, j, k]        # Predict
                elif ic == 1:
                    for j in range(n1):
                        for k in range(n3):
                            s[j, il, k] *= INV_SQRT2
                            s[j, ig, k] *= SQRT2
                            s[j, il, k] -= s[j, ig, k] * 0.5
                            s[j, ig, k] += s[j, il, k]
                else:
                    for j in range(n1):
                        for k in range(n2):
                            s[j, k, il] *= INV_SQRT2
                            s[j, k, ig] *= SQRT2
                            s[j, k, il] -= s[j, k, ig] * 0.5
                            s[j, k, ig] += s[j, k, il]
                
                il += step_incr
                ig += step_incr


