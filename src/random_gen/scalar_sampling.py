"""
A module with helper functions used to generate random scalar values from relevant distributions. 
"""

import numpy as np


def select_random_index(rng_main_, indices):
    """ Select 1 index at random from provided indices """
    # Ensure indices is an array-like list, otherwise it's a fuckery and rng_main_.choice returns a random number between 0 and indices if it contains only a 1 value. 

    # Convert scalar to list
    if np.isscalar(indices):
        indices = [int(indices)]
    else:
        indices = list(indices)
    if len(indices) == 1:
        return int(indices[0])
    
    return rng_main_.choice(indices, size=1, replace=False).astype(int)


def sample_uniform(min_value, max_value, rng_s, round_perc=0.):
    """ Sample from uniform - rounded at round_perc. """
    if min_value == max_value: 
        raise ValueError(f"min_value for pert of petro cannot be equal to max_value!")
    
    # If no rounding or round_perc is zero, return raw sample
    if round_perc <= 0.:
        return rng_s.uniform(min_value, max_value)
    elif round_perc>0.5:
        raise Exception("Wrong round_perc value. It is a ratio to define a percentage that should not be > 0.5")
    else: 
        # Round to nearest multiple of `round_to`
        round_to = round_perc * (max_value - min_value)
        return np.round(rng_s.uniform(min_value, max_value) / round_to) * round_to
    

def sample_gaussian(mu, sigma, rng_s,  round_perc=0.):
    """ Sample from normal distributio - to a percentage of sigma, round_perc. """
    if round_perc>0:
        round_to = round_perc * sigma
        return np.round(rng_s.normal(mu, sigma) / round_to) * round_to
    elif round_perc>0.5:
        raise Exception("Wrong round_perc value. It is a ratio to define a percentage that should not be > 0.5")
    else: 
        return rng_s.normal(mu, sigma)
    

def sample_pert_type(rng_main, config, callback_printer, exclude=None):
    """Sample perturbation type, optionally excluding certain types."""
    # TODO: with the definition of valid values: this could be made more efficient but giving the right set of values form the beginning.
    
    if exclude is None:
        exclude_set = set()
    elif isinstance(exclude, int):
        exclude_set = {exclude}
    else:
        exclude_set = set(exclude)  # Will raise TypeError if exclude is invalid

    valid_values = [i for i in range(*config.perts_all) if i not in exclude_set]

    if not valid_values:
        raise ValueError("No valid perturbation types left after exclusions")

    pert_type = rng_main.choice(valid_values)

    callback_printer(config.perturbation_messages.get(pert_type, "Invalid option!!!"))

    return pert_type


def get_perturbation_type(rng_main, config, n_births, n_births_max, log_run):
    """
    Get perturbation type, excluding birth perturbations if birth limit is exceeded.
    
    Args:
        rng_main: Random number generator
        perts_all: Range of perturbation types
        n_births: Current number of births
        n_births_max: Maximum allowed births
        pert_is_birth: Perturbation type that represents birth (typically 3)
        log_run: Logger
    
    Returns:
        int: Selected perturbation type
    """

    if n_births >= n_births_max-1:
        # Exclude birth. 
        exclude = config.pert_is_birth
        pert_type = sample_pert_type(rng_main, config, callback_printer=log_run.info, 
                                 exclude=exclude)
    else:
        pert_type = sample_pert_type(rng_main, config, callback_printer=log_run.info)
    return pert_type