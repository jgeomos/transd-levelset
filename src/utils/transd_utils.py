"""
Utility functions for nullspace and level set methods.

Provides core functionality for:
- Signed distance field calculations (level set representations)
- Petrophysical optimization 
- Birth/nucleation operations and gradient analysis
- Model derivation from signed distances
- Connected component analysis
- Unit contact calculations

These utilities support the main nullspace solver with geometric and
petrophysical operations for trans-dimensional geological inversion.
"""

import numpy as np
import skfmm
import warnings
import cc3d
from scipy.optimize import lsq_linear


def get_interface(phi_array: np.ndarray, index: int, tau: float) -> np.ndarray:
    """
    Compute a binary interface mask where |phi| < tau.

    Parameters
    ----------
    phi_array : np.ndarray -- signed distance field. 
    index : int, -- index of considered rock unit. 
    tau : float -- Threshold for determining interface: |phi| < tau.

    Returns
    -------
    np.ndarray -- Binary interface mask (0/1) with same shape as phi_array[index].
    """

    # Allocate mask array
    interface = np.zeros_like(phi_array[index])

    # Find interface locations
    interface_loc = np.where(np.abs(phi_array[index]) < tau)

    # Set points to 1 where interface criterion is met
    interface[interface_loc] = 1

    return interface

def get_petro_values(model_curr, return_counts=False):
    return np.unique(model_curr, return_counts=return_counts)


def get_rock_indices(signed_distances):
    """
    Takes in a np.array (dimension n_rocks, nx * ny * nz) signed_distances and returns an array of indices of signed_distances
    that has the highest value along the nx, ny, nz model coordinates. 

    :return: A numpy array containing the indices of rock units with highest signed distances.
    """
    return np.argmax(signed_distances, axis=0)


def get_density_model(petro_values_, signed_distances_):
    """
    Calculates the 3D petrophysical model from signed_distances_ for 
    rock unit petrophysical values given petro_values_.

    :param petro_values: an array of petrophysical values of rock units (1 value == value for 1 unit).
    :param signed_distances: a numpy array of signed distances for each rock unit.
    :return: a 1D numpy array of the same shape as the number of columns in signed_distances representing the
             calculated density model.
    """

    # Get the indices of the rock units corresponding to signed distances.
    rock_indices_ = get_rock_indices(signed_distances_)

    # Assign densities corresponding to
    # density_model_ = petro_values_[rock_indices_]

    return petro_values_[rock_indices_]


def nucl_calc_mask(mask, model, domains, grad_cost):
    """
    Determines nucleation locations based on gradient cost within specified domains and mask.

    Args:
        mask (np.ndarray): Mask selecting candidate nucleation cells (same shape as model).
        model (np.ndarray): Model array with domain labels (same shape as mask).
        domains (np.ndarray): Array of unique domain labels in model.
        grad_cost (np.ndarray): Gradient cost array (same shape as model).

    Returns:
        model_mask (np.ndarray): Binary mask (same shape as model) with 1 where nucleation can occur, 0 elsewhere.
        ind_nucl (tuple): Indices where nucleation can occur.
    """

    # Flatten arrays efficiently
    mask_flat = mask.ravel() if mask.ndim > 1 else mask
    model_flat = model.ravel() if model.ndim > 1 else model
    grad_flat = grad_cost.ravel() if grad_cost.ndim > 1 else grad_cost

    # Create masked model labels: set model label to -1 where mask is false (invalid)
    valid_cells = mask_flat > 0
    masked_model = np.where(valid_cells, model_flat, -1)

    # Compute sums of absolute gradient costs for each domain using nu.sums_gradient_cost
    sum_grad_cost = sums_gradient_cost(grad_flat, masked_model, domains)

    max_cost = sum_grad_cost.max() if sum_grad_cost.size > 0 else 0

    if max_cost == 0:
        empty_mask = np.zeros_like(model_flat)
        ind_nucl = (np.array([]),)
        return empty_mask.reshape(model.shape), ind_nucl

    best_domains = domains[sum_grad_cost == max_cost]

    # if len(best_domains) > 1:
    #     raise qc.NucleationError("PROBLEM: Cannot identify unique domain: multiple domains have equal max cost.")

    best_domain = best_domains[0]

    # Create nucleation mask: where model equals best_domain and cell is valid
    nucleation_mask = (model_flat == best_domain) & valid_cells

    ind_nucl = np.where(nucleation_mask)

    return nucleation_mask.astype(float), ind_nucl


def get_birth_value(model_class, pert_dens_nucl, birth_params):

    # Create trial model
    if model_class.m_tmp is None: 
        model_class.m_tmp = np.empty_like(model_class.m_curr)
    np.copyto(model_class.m_tmp, model_class.m_curr)

    # Extract the birthed value. TODO is this really necessary?
    petro_pre_birth = get_petro_values(model_class.m_tmp)
    dens_mod_update(model_class.m_tmp, pert_dens_nucl, birth_params.ind_nucl, model_class.m_curr.shape)
    petro_post_birth = get_petro_values(model_class.m_tmp)
    new_value_birth = find_new_petro(petro_post_birth, petro_pre_birth)

    return new_value_birth


def calc_gradient_connected_regions(grad_cost, birth_params, j):
    """
    Calculate data misfit gradient and find connected regions above threshold.
    
    Args:
        sensit: Sensitivity matrix
        geophy_data: Geophysical data
        gradient_threshold: Threshold for gradient selection
        N_blobs: Number of connected regions to find
        dim: Model dimensions
        
    Returns:
        tuple: (grad_cost, mask_grad_labels, N) where grad_cost is the gradient,
               mask_grad_labels are the labeled connected regions, and N is the number of regions
    """

    # Scale with depth weighting (TODO: TEST IT)
    # grad_cost *= sensit.weight

    # Get cells with gradient superior to threshold and set them to 1, with 0 for the rest.
    max_grad_ind = get_high_grad(grad_cost, birth_params.grad_thresh[j])

    # Fill maks with 0 where gradient of cost lower than threshold, and one otherwise. 
    birth_params.mask_max_grad.fill(0)
    birth_params.mask_max_grad.flat[max_grad_ind] = 1 
    
    # Group N_blobs contiguous 'clumps' of cells with a gradient superior to selected threshold.
    mask_grad_labels, birth_params.n_blobs_curr = get_connected_cells(birth_params.mask_max_grad, k=birth_params.n_blobs_max)
    
    return mask_grad_labels, birth_params.n_blobs_curr


def find_new_petro(new_petro_array, previous_petro_array):
    """
    Return element from `new_petro_array` that are not present in `previous_petro_array`.

    Args:
        new_petro_array (iterable): Sequence of candidate or updated values.
        previous_petro_array (iterable): Sequence of baseline or already existing values.

    Returns:
        np.ndarray: Values in `new_petro_array` not present in `previous_petro_array`.
    """
    return np.array([value for value in new_petro_array if value not in previous_petro_array])


def get_high_grad(gradient_cost, gradient_threshold):
    """ Find cells with  gradient_cost > gradient_threshold. """
    
    abs_gradient = np.abs(gradient_cost)
    max_val = np.max(abs_gradient)
    threshold = gradient_threshold * max_val
    
    # Use flatnonzero for better performance on large arrays
    return np.flatnonzero(abs_gradient > threshold)


def calc_petro_opti(response_matrix, residuals_vector, bounds_constraints, max_iterations=100, verbose=False):
    """
    Solves: min ||A*x - b||² subject to bounds, where:
    - A (response_matrix): Forward model responses for each unit
    - b (residual_vector): Data residuals (observed - calculated)
    - x: Optimal petrophysical perturbations (returned)
    
    Args:
        response_matrix: Forward responses, shape (n_data, n_units) or (n_data,)
        residual_vector: Data residuals, shape (n_data,)
        bounds: Bounds for optimization

    Solve the inverse problem to obtain the best single petrophysical property value for rock units taken as group to reduce misfit.

    Calculate x, the best multiplying coefficient to reducing misfit based on uniform petrophysical values in each unit.
    Solve least squares system of equation to fit the measurements by linear combination of contributions of the
    different rock unit's response

    A = matrix where the i-th column corresponds to the fwd data for i-th rock unit in the current model.
    b = residuals of data, d_obs - d_calc.
    For the optimisation of n rock unit petrophysical values, A has n columns and x with have n elements.

    """

    residuals_vector = np.squeeze(residuals_vector)

    # Ensure response_matrix is 2D (n_data, n_units).
    if response_matrix.ndim == 1:
        response_matrix = response_matrix.reshape(-1, 1)
    elif response_matrix.ndim != 2:
        raise ValueError(f"response_matrix must be 1D or 2D, got {response_matrix.ndim}D")
    
    # Solve bounded least squares.
    petro_opti = lsq_linear(response_matrix, residuals_vector, bounds=bounds_constraints,
                       max_iter=max_iterations, verbose=int(verbose))

    return petro_opti


def sums_gradient_cost(grad_flat, model_mask, domains):
    """
    Compute the sum of absolute gradient values for each domain.

    Parameters
    ----------
    grad_flat : np.ndarray
        Flattened gradient values.
    model_mask : np.ndarray
        Array of same shape as grad_flat, containing domain labels.
    domains : array-like
        List or array of domains to compute sums for.

    Returns
    -------
    np.ndarray
        Array of summed absolute gradient values for each domain, 
        in the same order as `domains`.
    """
    abs_grad = np.abs(grad_flat)

    # Case 1: domains are integers (fast path)
    if np.issubdtype(model_mask.dtype, np.integer):
        sums = np.bincount(model_mask, weights=abs_grad)
        return np.array([sums[d] if d < len(sums) else 0 for d in domains])

    # Case 2: fallback for arbitrary labels
    return np.array([abs_grad[model_mask == domain].sum() for domain in domains])


def dens_mod_update(model_array, pert_dens_nucleate, ind_nucl, dim_model=None):
    """
    For births: adds density perturbation to specified indices.
    Always works with flattened view for consistency.
    """
    # Use flat view for perf. 
    model_flat = model_array.ravel() if model_array.ndim > 1 else model_array
    model_flat[ind_nucl] += pert_dens_nucleate

    if dim_model is not None and model_array.ndim == 1:
        return model_array.reshape(dim_model)
    return model_array


def get_connected_cells(mask_max_grad, k):
    """
    Identifies k groups (clumps) of contiguous cells in mask_max_grad with the values of 1. 
    :param mask_max_grad: 0 for values of gradient lower than threshold of prev step. 1 for values superior. 
    :param k: number of desired clumps of connected cells to consider - the kth largest
    :return: volume mask with labels or ordered clumps of n connected cells
    
    This function uses the algorithm from: 
    Silversmith, W. (2021). cc3d:
    Connected components on multilabel 3D & 2D images. (Version 3.2.1) [Computer software].
    https://doi.org/https://zenodo.org/record/5535251
    https://github.com/seung-lab/connected-components-3d/
    """

    return cc3d.largest_k(mask_max_grad, k=k, connectivity=26, delta=0, return_N=True)


# ======================== Death utils. 
def get_smallest_birthed_units(petrovals, mvars, n=2):
    """
    Get the n smallest birthed units (by cell count).
    
    Parameters
    ----------
    petrovals : PetroStateManager
        Petrophysical state manager with tracking
    mvars : ModelStateManager
        Model state with current model
    n : int, default=2
        Number of smallest units to return
        
    Returns
    -------
    list of tuples : [(index, value, count, item_id), ...]
        Each tuple contains:
        - index: current sorted index in tracked array
        - value: petrophysical value
        - count: number of cells
        - item_id: unique identifier
        Returns empty list if no birthed units found.
        
    Examples
    --------
    >>> # Get two smallest birthed units
    >>> smallest = get_smallest_birthed_units(petrovals, mvars, n=2)
    >>> if len(smallest) == 2:
    >>>     ind1, val1, count1, id1 = smallest[0]
    >>>     ind2, val2, count2, id2 = smallest[1]
    >>> elif len(smallest) == 1:
    >>>     ind, val, count, item_id = smallest[0]
    """
    # Get values and counts from current model
    values, counts = get_petro_values(mvars.m_curr, return_counts=True)
    count_dict = dict(zip(values, counts))
    
    # Collect birthed units with their counts
    birthed_units = []
    for i, (val, item_id, origin, _, _, _) in enumerate(petrovals._tracked_state["items"]):
        if origin == 'birth':
            count = count_dict.get(val, 0)
            birthed_units.append((i, val, count, item_id))
    
    # Sort by count (ascending) to get smallest first
    birthed_units.sort(key=lambda x: x[2])
    
    # Return up to n smallest
    return birthed_units[:n]


def calculate_unit_contacts_fast(model, cell_sizes=None):
    """
    Fast vectorized calculation of contact surface areas.
    """
    if cell_sizes is None:
        cell_sizes = (1.0, 1.0, 1.0)
    dx, dy, dz = cell_sizes
    
    contacts = {}
    
    # X-direction contacts
    diff_x = model[:-1, :, :] != model[1:, :, :]
    if np.any(diff_x):
        pairs_x = np.stack([model[:-1, :, :][diff_x], model[1:, :, :][diff_x]], axis=1)
        area_x = dy * dz
        for pair in np.unique(pairs_x, axis=0):
            key = tuple(sorted(pair))
            count = np.sum((pairs_x == pair).all(axis=1))
            contacts[key] = contacts.get(key, 0) + count * area_x
    
    # Y-direction contacts
    diff_y = model[:, :-1, :] != model[:, 1:, :]
    if np.any(diff_y):
        pairs_y = np.stack([model[:, :-1, :][diff_y], model[:, 1:, :][diff_y]], axis=1)
        area_y = dx * dz
        for pair in np.unique(pairs_y, axis=0):
            key = tuple(sorted(pair))
            count = np.sum((pairs_y == pair).all(axis=1))
            contacts[key] = contacts.get(key, 0) + count * area_y
    
    # Z-direction contacts
    diff_z = model[:, :, :-1] != model[:, :, 1:]
    if np.any(diff_z):
        pairs_z = np.stack([model[:, :, :-1][diff_z], model[:, :, 1:][diff_z]], axis=1)
        area_z = dx * dy
        for pair in np.unique(pairs_z, axis=0):
            key = tuple(sorted(pair))
            count = np.sum((pairs_z == pair).all(axis=1))
            contacts[key] = contacts.get(key, 0) + count * area_z
    
    return contacts


def calculate_contacts_for_unit(model, target_unit, cell_sizes=None, mode="all"):
    """
    Efficiently calculate contact areas only for a specific target unit.
    Works with any dtype labels (e.g., floats, ints).

    Parameters
    ----------
    model : np.ndarray
        3D labeled model array
    target_unit : scalar
        Unit value to compute contacts for
    cell_sizes : tuple
        Dimensions (dx, dy, dz)
    mode : str
        "all"     -> return full dictionary {other_unit: area}
        "largest" -> return only (unit, area) tuple
        "top_two" -> return {'largest': (...), 'second_largest': (...)}

    Returns
    -------
    dict or tuple
    """
    if cell_sizes is None:
        cell_sizes = (1.0, 1.0, 1.0)
    dx, dy, dz = cell_sizes

    contacts = {}

    def accumulate_contacts(neighbor_units, area_per_contact):
        """Helper to accumulate contact areas."""
        if neighbor_units.size > 0:
            units, counts = np.unique(neighbor_units, return_counts=True)
            for u, c in zip(units, counts):
                if u != target_unit:  # skip self-contact
                    contacts[u] = contacts.get(u, 0.0) + c * area_per_contact

    # X-direction faces
    left = model[:-1, :, :]
    right = model[1:, :, :]
    accumulate_contacts(right[(left == target_unit) & (right != target_unit)], dy * dz)
    accumulate_contacts(left[(right == target_unit) & (left != target_unit)], dy * dz)

    # Y-direction faces
    front = model[:, :-1, :]
    back = model[:, 1:, :]
    accumulate_contacts(back[(front == target_unit) & (back != target_unit)], dx * dz)
    accumulate_contacts(front[(back == target_unit) & (front != target_unit)], dx * dz)

    # Z-direction faces
    bottom = model[:, :, :-1]
    top = model[:, :, 1:]
    accumulate_contacts(top[(bottom == target_unit) & (top != target_unit)], dx * dy)
    accumulate_contacts(bottom[(top == target_unit) & (bottom != target_unit)], dx * dy)

    # If no specific summary requested, return full dictionary
    if mode == "all":
        return contacts

    # If no contacts found and summary requested
    if not contacts:
        if mode == "largest":
            return (None, 0.0)
        elif mode == "top_two":
            return {'largest': (None, 0.0), 'second_largest': (None, 0.0)}
        # TODO add error if not contacts.

    # Sort contacts by descending area
    sorted_contacts = sorted(contacts.items(), key=lambda x: x[1], reverse=True)

    if mode == "largest":
        return sorted_contacts[0]  # tuple (unit, area)

    if mode == "top_two":
        largest = sorted_contacts[0]
        second_largest = sorted_contacts[1] if len(sorted_contacts) > 1 else (None, 0.0)
        return {'largest': largest, 'second_largest': second_largest}

    raise ValueError(f"Invalid mode '{mode}'. Use 'all', 'largest', or 'top_two'.")


# ======================== Signed Distance Operations
def set_inf_signdist_values(petro_voxet, signdist, unit_index, petro_val):
    """
    Assign +inf/-inf sign dist values to a selected rock unit from petro_val_vector with unit_index.
    Looks in the petro_voext where the unit is, and uses this info to set the correspond signed distances 
    to +/- inf values to make sure it is not modified at the next stage of the workflow. 
    """

    dimensions_signdist = np.shape(signdist)
    dimensions_petro_voxet = np.shape(petro_voxet)

    if len(dimensions_petro_voxet) != 3:
        # raise "'dimension' should correspond to a 3D array!"
        shape_voxet = dimensions_signdist[1:]
        petro_voxet = petro_voxet.reshape(shape_voxet)

    ind = np.where(petro_voxet == petro_val)
    signdist[unit_index, ind[0], ind[1], ind[2]] = + np.inf

    # Set the sign dist value of anomaly to large negative dummy value to be sure it does not grow. 
    ind = np.where(petro_voxet != petro_val)
    signdist[unit_index, ind[0], ind[1], ind[2]] = - np.inf

    return signdist


def get_indices_largest_signdist(signed_distances, n_largest_values):
    """
    Get the indices of the `n_largest_values` largest values in a signed distance array.

    This function identifies the `n_largest_values` elements with the highest values in
    the `signed_distances` array and returns their corresponding indices.

    Parameters:
    -----------
    signed_distances : np.ndarray, A NumPy array containing signed distance values.
    n_largest_values : int, The number of highest-value elements to retrieve.

    Returns:
    --------
    tuple of np.ndarray
        A tuple of NumPy arrays representing the multi-dimensional indices of the
        `num_live_cells` largest values in `signed_distances`.
    """

    flat_indices = np.argpartition(signed_distances.flatten(), -n_largest_values)[-n_largest_values:]
    sorted_flat_indices = flat_indices[np.argsort(signed_distances.flatten()[flat_indices])[::-1]]
    largest_values = np.unravel_index(sorted_flat_indices, signed_distances.shape)

    return largest_values


def update_signed_distance_all(phi_ls, d_phi_ls):
    """
    Update signed distance fields phi_ls with scalar fields d_phi_ls.
    """
    assert phi_ls.shape == d_phi_ls.shape, f"Arrays have different shapes: {phi_ls.shape} vs {d_phi_ls.shape}"

    # Mask where updates are non-zero
    mask = d_phi_ls != 0

    # Add perturbation update. 
    phi_ls[mask] += d_phi_ls[mask]

    return phi_ls


def calc_signed_distances(mod_dens_const, drho0, cell_size=[1, 1, 1], narrow=False):
    """
    Calculates the signed distances from the given parameters using the Fast Marching Method (FMM).

    Parameters:
    --------------
    :param mod_dens_const: array_like, density model, 3D matrix.
        Discrete density contrast model
    :param drho0 : array_like
        Density contrast vector (gm/m^3)
    :param cell_size : array_like
        Physical dimension of model cells, optional (default is [1, 1, 1])
    :param narrow : bool
        Narrow band flag, optional (default is False)

    :return:
    phi : array_like.
        Array of the signed distances after calculation using the FMM
    """

    assert mod_dens_const.ndim == 3, "Input model array should not be flat or 2D. It should be a 3D."

    signdist = np.zeros((drho0.size, mod_dens_const.shape[0], mod_dens_const.shape[1], mod_dens_const.shape[2]), 
                        dtype=np.float32)

    # Loop over each density contrast value.
    for i in range(0, drho0.size):

        if drho0.size == 1:

            signdist[i][mod_dens_const != drho0] = -1
            signdist[i][mod_dens_const == drho0] = 1

        else: 
            signdist[i][mod_dens_const != drho0[i]] = -1
            signdist[i][mod_dens_const == drho0[i]] = 1
        # Use FMM with narrow or wide banding.
        # arg. 'periodic' controls boundary conditions.
        if narrow:
            result = skfmm.distance(signdist[i], cell_size, order=2, periodic=False, narrow=2 * cell_size[0])
            print('Calculating BAND LIMITED signed-dist. (2*dim[0] band)')
        else:
            result = skfmm.distance(signdist[i], cell_size, order=2, periodic=False)

        signdist[i] = result.astype(np.float32)

    return signdist


def unmask_signdist(phi_masked, phi_for_approx):
    """
    When using narrow band FMM, absence of values away from interfaces is masked with -inf values.
    This function replaces these values from phi_masked with values from an approximation in phi_for_approx (can be from
    the previous iteration).

    # TODO: JG introduced modif in the FMM function about mask -> have it here with proper ref instead?

    :param phi_masked:
    :param phi_for_approx:
    :return:
    """

    mask = phi_masked == False
    phi_masked[mask] = phi_for_approx[mask]


def compute_signed_distance(signdist_i_, cell_size, narrow):
        if narrow:
            result = skfmm.distance(signdist_i_, cell_size, order=1, periodic=False, narrow=2 * cell_size[0])
        else:
            result = skfmm.distance(signdist_i_, cell_size, order=1, periodic=False)
        return result.astype(np.float32) 


def calc_signed_distances_opti(mod_dens_new, mod_dens_ref, phi_ref, drho0, cell_size=[1, 1, 1], 
                               birth=False, death=False, narrow=False, nprocs=1, log_run=None):
    """ 
    Partially optimised calculation of signed distances to interfaces.
    Relies on narrow band calculation, splitting the model in submodels 
    and identification of differences with reference model.

    """

    # TODO MAKE SURE RESULTS WITH THIS APPROACH ARE EXACTLY THE SAME AS OTHERWISE. 
    # TODO have a slight overlap between chuncks to mitigate border effects? 
    # TODO Process each part sequentially -- TODO what could do is split accordingly with where there's something to calcualte e.g. the mask to reduce the calc space further. 

    assert mod_dens_new.ndim == 3, "Input model array should not be flat or 2D. It should be a 3D."
    assert mod_dens_ref.ndim == 3, "Input model array should not be flat or 2D. It should be a 3D."
    if birth and death: 
        raise ValueError("Cannot have both birth and death!")

    # Create masks for differences and matches.
    difference_mask = mod_dens_new != mod_dens_ref

    # Collect values at differing locations from both voxets.
    values_voxet1 = mod_dens_new[difference_mask]
    values_voxet2 = mod_dens_ref[difference_mask]

    # Combine and extract the unique values involved in mismatches.
    mismatched_values = np.unique(np.concatenate([values_voxet1, values_voxet2]))

    new_drho0 = drho0.copy()

    # Case of a Birth: Find values from mismatched_values that are not in drho0
    if birth: 
        new_drho0 = get_petro_values(mod_dens_new)
        mask = np.isin(new_drho0, drho0)
        # indices_common_petro = np.where(mask)[0]
        drho0 = get_petro_values(mod_dens_ref)

    if mismatched_values.size == 0:
        return phi_ref.astype(np.float32) 

    else: 
        n_rocks = new_drho0.shape[0]
        shp_model = np.shape(phi_ref)[1:]
        signdist = np.zeros((n_rocks, shp_model[0], shp_model[1], shp_model[2]),
                            dtype=np.float32)

        # Get correspondance between new_drho0 and drho0.
        # Build a value-to-index map from for drho0 to identify units not present in one but present in the other.
        value_to_index = {value: idx for idx, value in enumerate(drho0)}
        indices_rocks = [value_to_index.get(val, -1) for val in new_drho0]  # returns -1 if value not found

        if log_run is not None: 
            log_run.info(f"indices_rocks for calc of sign dist.: {indices_rocks}")

        ind_diff = []
        for i in range(0, new_drho0.size):
            if indices_rocks[i] < phi_ref.shape[0]:
                signdist[i] = phi_ref[indices_rocks[i]]
            if birth: 
                narrow_tmp = False 
            else:
                narrow_tmp = narrow
            if new_drho0[i] in mismatched_values:
                ind_diff.append(i)
                mask = (mod_dens_new == new_drho0[i])
                signdist[i] = np.where(mask, 1.0, -1.0).astype(np.float32)
                
                #--- Split the domain into 2×2×2 = 8 parts to reduce the cost, which goes as nlog(n).
                # Calculate split points for each dimension
                mid_x = shp_model[0] // 2
                mid_y = shp_model[1] // 2
                mid_z = shp_model[2] // 2

                for x_idx in range(2):
                    for y_idx in range(2):
                        for z_idx in range(2):  # TODO here add a bit of overlap between chunks, e.g. a few cells.
                            # Calculate slice indices
                            x_start = x_idx * mid_x
                            x_end = shp_model[0] if x_idx == 1 else mid_x
                            
                            y_start = y_idx * mid_y
                            y_end = shp_model[1] if y_idx == 1 else mid_y
                            
                            z_start = z_idx * mid_z
                            z_end = shp_model[2] if z_idx == 1 else mid_z

                            mask_part = mask[x_start:x_end, y_start:y_end, z_start:z_end]

                            # Process the current part making a subset of the full model.
                            if np.all(mask_part):  # All True - set to +1
                                signdist[i][x_start:x_end, y_start:y_end, z_start:z_end] = False
                            elif np.all(~mask_part):  # All False - set to -1
                                signdist[i][x_start:x_end, y_start:y_end, z_start:z_end] = False
                            else:  # Mixed True/False - compute signed distance
                                signdist[i][x_start:x_end, y_start:y_end, z_start:z_end] = compute_signed_distance(signdist[i][x_start:x_end, y_start:y_end, z_start:z_end], 
                                                                                                                   cell_size, narrow_tmp)
                
                # Convert to MaskedArray if needed
                # signdist[i] = np.ma.MaskedArray(signdist[i], mask)

        #- Run things in parallel.
        # results = Parallel(n_jobs=nprocs, backend='threading')(
        #     delayed(compute_signed_distance)(signdist[j].astype(int), cell_size, narrow)
        #     for j in ind_diff
        #     )
        # results = np.array(results)
        # Assign results back to the correct indices
        # for idx, j in enumerate(ind_diff):
        #     signdist[j] = results[idx]

    if narrow:
        warnings.warn("Be sure this does not collide with the perturbation of sign distances!")
        warnings.warn("Calc. BAND LIMITED signed-dist (2*dim[0] band). In pfmm.py, this version uses JG modif to calc of mask!")
        for i in range(0, new_drho0.size):
            if indices_rocks[i] != -1:
                # TODO un-mask only the units accordingly with births. 
                unmask_signdist(signdist[i], phi_ref[indices_rocks[i]])

    return signdist.astype(np.float32)

