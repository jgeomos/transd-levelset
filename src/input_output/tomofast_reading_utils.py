"""
Functions to read data & model grids from Tomofast-x format.
For more information: https://github.com/TOMOFAST/Tomofast-x
"""

import numpy as np
import os
from scipy.sparse import csr_matrix
import warnings
warnings.simplefilter("always", category=UserWarning)  # Make sure warnings are actually printed when using Jupyter Notebook.


def load_sensit_from_tomofastx(sensit_path, nbproc, empty_sensit_class, type="grav", verbose=False, path=None):
    """
    Loads the sensitivity kernel from Tomofast-x and stores it in the CSR sparse matrix.
    """

    if path is not None: 
        sensit_path = path + "\\" + sensit_path

    if type == "grav":
        prefix_sensit_name = "sensit_grav_"
    elif type == "magn":
        prefix_sensit_name = "sensit_magn_"
    else:
        if verbose: 
            print(type)
        raise Exception('Wrong type of sensitivity matrix!')

    # Metadata file.
    filename_metadata = sensit_path + "/" + prefix_sensit_name + "meta.txt"
    # Depth weight file.
    filename_weight = sensit_path + "/" + prefix_sensit_name + "weight"

    #----------------------------------------------------------
    # Reading the metadata.
    with open(filename_metadata, "r") as f:
        lines = f.readlines()

        # Read model dimensions.
        nx = int(lines[0].split()[0])
        ny = int(lines[0].split()[1])
        nz = int(lines[0].split()[2])

        # Reading the number of data.
        ndata_read = int(lines[0].split()[3])

        if verbose:
            print('Tomofastx nx, ny, nz =', nx, ny, nz)
            print('ndata_read =', ndata_read)

        # Reading the number of procs.
        nbproc_read = int(lines[1].split()[0])

        if (nbproc != nbproc_read):
            raise Exception('Inconsistent nbproc!')

        if verbose:
            print('nbproc_read =', nbproc_read)

        compression_type = int(lines[2].split()[0])

        if verbose:
            print('compression_type =', compression_type)

        if compression_type > 1:
            raise Exception('Inconsistent compression type!')

        # The number of non-zero values.
        nnz_total = int(lines[4].split()[0])

        if verbose:
            print("nnz_total =", nnz_total)

    #----------------------------------------------------------
    # Reading depth weight.
    nel_total = nx * ny * nz
    with open(filename_weight, "r") as f:
        # Note using '>' for big-endian.
        header = np.fromfile(f, dtype='>i4', count=1)
        weight = np.fromfile(f, dtype='>f8', count=nel_total)

    #----------------------------------------------------------
    # Define spase matrix data arrays.
    # Note we a matrix constructor where the csr_row stores row indexes of all elements: a[row_ind[k], col_ind[k]] = data[k].
    csr_dat = np.ndarray(shape=(nnz_total), dtype=np.float32)
    csr_row = np.ndarray(shape=(nnz_total), dtype=np.int32)
    csr_col = np.ndarray(shape=(nnz_total,), dtype=np.int32)

    nel_current = 0
    ndata_all = 0

    # Loop over parallel matrix chunks.
    for n in range(nbproc):

        # Sensitivity kernel file.
        filename_sensit = sensit_path + "/" + prefix_sensit_name + str(nbproc) + "_" + str(n)

        # Building the matrix arrays.
        with open(filename_sensit, "r") as f:
            # Global header.
            header = np.fromfile(f, dtype='>i4', count=5)
            ndata_loc = header[0]
            ndata = header[1]
            nmodel = header[2]

            if verbose:
                print("ndata_loc =", ndata_loc)
                print("ndata =", ndata)
                print("nmodel =", nmodel)

            ndata_all += ndata_loc

            # Loop over matrix rows.
            for i in range(ndata_loc):
                # Local line header.
                header_loc = np.fromfile(f, dtype='>i4', count=4)

                # Global data index.
                idata = header_loc[0]

                # Number of non-zero elements in this row.
                nel = header_loc[1]

                # Reading one matrix row.
                col = np.fromfile(f, dtype='>i4', count=nel)
                dat = np.fromfile(f, dtype='>f4', count=nel)

                # Array start/end indexes corresponding to the current matrix row.
                s = nel_current
                e = nel_current + nel

                csr_col[s:e] = col
                csr_row[s:e] = idata - 1
                csr_dat[s:e] = dat

                nel_current = nel_current + nel
    #----------------------------------------------------------
    if verbose:
        print('ndata_all =', ndata_all)

    if (ndata_all != ndata_read):
        raise Exception('Wrong ndata value!')

    # Shift column indexes to convert from Fortran to Python array index.
    csr_col = csr_col - 1

    # Convert units from Tomofast to geomos (as we use different gravitational constant).
    if type == 'grav':
        csr_dat = csr_dat * 1.e+3  # TODO: make it all SI.

    # Create a sparse matrix object.
    matrix = csr_matrix((csr_dat, (csr_row, csr_col)), shape=(ndata_all, nmodel))

    sensit = empty_sensit_class(nx, ny, nz, compression_type, matrix, weight)

    # Keep minimal verbose.
    print("Sensitivity matrix from Tomofastx: loaded.")

    return sensit


def read_tomofast_data(geophy_data, filename, data_type, path=None):
    """
    Read data and grid stored in Tomofast-x format.
    """

    if filename is None:
        warnings.warn(f"File {filename} not found --> Not loaded!!", UserWarning)
        if data_type=='field':
            raise Exception('A data is mandatory, yet incorrectly provided!')
        elif data_type=='background':
            geophy_data.background = None
        else: 
            raise Exception(f'Unsupported data_type: {data_type} was provided!')
        return None

    if path is not None: 
        filename = path  + "\\" + filename

    data = np.loadtxt(filename, skiprows=1)

    if data_type == 'field':
        geophy_data.data_field = data[:, 3]

    elif data_type == 'background':
        geophy_data.background = data[:, 3]

    # Reading the data grid.
    geophy_data.x_data = data[:, 0]
    geophy_data.y_data = data[:, 1]
    geophy_data.z_data = data[:, 2]


def read_tomofast_model(filename, mpars, path=None):
    """
    Read model values and model grid stored in Tomofast-x format.
    Stores the coordinates of model cells in the class instance mpars.
    """

    if filename is None:
        warnings.warn(f"File {filename} not found --> Not loaded!!", UserWarning)
        return None, None

    if path is not None: 
        filename = path  + "\\" + filename

    # Check if the file exists.
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    else:
        with open(filename, "r") as f:
            lines = f.readlines()
            n_elements = int(lines[0].split()[0])

        # Sanity check.
        assert n_elements == np.prod(mpars.dim), (
                                                f"Mismatching between input file and grid class! "
                                                f"n_elements={n_elements}, expected={np.prod(mpars.dim)}"
                                                )

        model = np.loadtxt(filename, skiprows=1)

        # Get model values.
        m_inv = model[:, 6]
        m_inv = m_inv.reshape(mpars.dim)

        # Define model grid (cell edges).
        mpars.x1 = model[:, 0]
        mpars.x2 = model[:, 1]
        mpars.y1 = model[:, 2]
        mpars.y2 = model[:, 3]
        mpars.z1 = model[:, 4]
        mpars.z2 = model[:, 5]

        # Define model grid (cell centers).
        mpars.x = 0.5 * (mpars.x1 + mpars.x2)
        mpars.y = 0.5 * (mpars.y1 + mpars.y2)
        mpars.z = 0.5 * (mpars.z1 + mpars.z2)

        # Convert to km.
        mpars.x = mpars.x / 1000.
        mpars.y = mpars.y / 1000.
        mpars.z = mpars.z / 1000.

        return m_inv, mpars
