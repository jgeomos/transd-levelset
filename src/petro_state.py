"""
Petrophysical state management for inversion.

This module provides PetroStateManager, a class for tracking petrophysical
properties across units during inversion, including handling
of births, deaths, and modifications with full history tracking.

Authors:
    Jérémie Giraud
    Vitaliy Ogarko

License: MIT
"""

__email__ = "jeremie.giraud@uwa.edu.au"


import numpy as np
import src.utils.quality_control as qc
import src.utils.transd_utils as tu


class PetroStateManager:
    """
    Manages petrophysical property state with change tracking.
    
    Tracks evolution of petrophysical values across geological units during
    inversion, supporting birth (insertion), death (removal), and modification
    operations with full history and reversibility.
    
    The class maintains:
    - Current values for all units
    - Original (initial) values
    - Previous values for revert operations
    - Origin tracking (original vs birthed)
    - Unique identifiers for each unit
    
    Attributes
    ----------
    all_values : ndarray
        Current petrophysical values (sorted)
    orig_values : ndarray
        Original petrophysical values
    petro_prev : ndarray
        Previous petrophysical values (for revert)
    n_units_total : int
        Total number of units
    n_units_original : int
        Number of original units
    n_units_birthed : int
        Number of birthed units
    
    Examples
    --------
    >>> # Initialize from model
    >>> petrovals = PetroStateManager(
    ...     model_curr=initial_model,
    ...     model_with_pert=anomaly_model
    ... )
    >>> 
    >>> # Birth a new unit
    >>> new_id = petrovals.insert_value(2650.0)
    >>> 
    >>> # Modify a unit
    >>> petrovals.modify_value_by_index(2, 2700.0)
    >>> 
    >>> # Remove a birthed unit
    >>> petrovals.remove_value(item_id=new_id)
    >>> 
    >>> # Check composition
    >>> print(f"Total: {petrovals.n_units_total}")
    >>> print(f"Has births: {petrovals.has_birthed_units()}")
    
    See Also
    --------
    ModelStateManager : Manages geometric model state
    InversionMetrics : Tracks inversion metrics and convergence
    """

    # Define allowed attributes.
    __slots__ = ['_tracked_state', 'std_petro', 'distro_type', 'pert']  
   
    def __init__(self, std_petro=5., distro_type='gaussian', pert=np.array([0, 0]), 
                model_curr=None, model_with_pert=None, log_run=None, 
                track_histograms=False, histogram_bins=50, histogram_range=None):
        """
        Initialize PetroStateManager with petrophysical properties and optional model-based initialization.
        
        Parameters:
        -----------
        std_petro : float, default=5.
            Standard deviation for petro values uncertainty
        distro_type : str, default='gaussian'
            Type of distribution ('uniform' or 'gaussian')
        pert : np.ndarray, default=[0, 0]
            Perturbation vector [background, anomaly] - used as fallback if models not provided
        model_curr : optional
            Current model for extracting initial petrophysical values
        model_with_pert : optional  
            Model with perturbations for calculating pert values
        log_run : optional
            Logger instance for verbose output
        """
        # Tracked state for sophisticated array management
        self._tracked_state = None
        
        # Basic parameters
        self.std_petro: float = std_petro
        self.distro_type: str = distro_type
        self.pert: np.ndarry = pert
        
        # Validation checks for distro_type
        if self.distro_type not in ('uniform', 'gaussian'):
            raise qc.ParameterValidationError(f"Invalid distribution type: {self.distro_type}. Must be 'uniform' or 'gaussian'")
        
        # Initialize with models if provided, otherwise use default pert
        if model_curr is not None and model_curr.size != 0:
            # Get initial values and initialize tracked array
            initial_values = tu.get_petro_values(model_curr)

            # Initialise the petrophysical dictionnar with tracking of property evolution.
            self.init_tracked_array(initial_values)
        else: 
            raise qc.ModelValueError(f"Invalid value for model_curr: {model_curr}. It should be a model!")

        if model_with_pert is not None and model_curr.size != 0:
            
            # Calculate perturbation from provided model.
            self.set_pert_from_model(model_with_pert)

            # Validation checks for pert.
            if not isinstance(self.pert, np.ndarray):
                raise qc.ParameterValidationError("`pert` must be a numpy array.")
            
            if len(self.pert) != 2:
                raise qc.ParameterValidationError(
                    f"`pert` must have exactly 2 elements (background, anomaly), got {len(self.pert)} elements: {self.pert}\n"
                    f"  Check perturbation_filename in your parfile points to a valid model\n"
                )
            
        # Logging and verbose output.
        if log_run is not None: 
            self._log_or_print(f'petrovals.pert = {self.pert}', log_run)
            self._log_or_print(f'petrovals.all_values = {self.all_values}', log_run)
            self._log_or_print('Vector of petrophysical values: initialised with tracked array.', log_run)
            self._log_or_print('', log_run)

    def check_units_health(self, model_curr, min_cells_birthed=100, protect_original=True):
        """
        Get indices of units that should be considered for death.
        
        Parameters
        ----------
        model_curr : np.ndarray
            Current model
        min_cells_original : int
            Death threshold for original units
        min_cells_birthed : int
            Death threshold for birthed units
        protect_original : bool
            If True, never mark original units for death
            
        Returns
        -------
        list : Indices of units that are candidates for death
        """
        values, counts = tu.get_petro_values(model_curr, return_counts=True)    # This line creates a dependency. Should this be removed? 
        count_dict = dict(zip(values, counts))
        
        should_kill_smallest = False
        
        if not self._tracked_state:
            return should_kill_smallest, -1, -1, -1
        
        for i, (val, item_id, origin, _, _, _) in enumerate(self._tracked_state["items"]):
            count = count_dict.get(val, 0)
            
            # Skip original units if protected
            if protect_original and origin == 'orig':
                continue
            
            # Use appropriate threshold
            if count < min_cells_birthed:
                should_kill_smallest = True
                return should_kill_smallest, i, val, item_id
        
        return should_kill_smallest, -1, -1, -1
    
    @property
    def n_units_total(self):
        """Total number of units."""
        if self._tracked_state is None:
            return 0
        return len(self._tracked_state["items"])
    
    def get_unit_statistics(self, model_curr):
        """
        Get detailed statistics about all units.
        
        Returns
        -------
        pd.DataFrame or dict : Statistics for each unit including origin, size, etc.
        """
        values, counts = tu.get_petro_values(model_curr, return_counts=True)
        count_dict = dict(zip(values, counts))
        
        stats = []
        total_cells = model_curr.size
        
        for i, (val, item_id, origin, orig_val, _, orig_idx) in enumerate(self._tracked_state["items"]):
            count = count_dict.get(val, 0)
            stats.append({
                'index': i,
                'value': val,
                'origin': origin,
                'original_value': orig_val,
                'original_index': orig_idx,
                'count': count,
                'percentage': (count / total_cells) * 100,
                'is_alive': count > 0
            })
        
        return stats

    def _log_or_print(self, message, log_run=None):
        """Centralized logging/printing."""
        if log_run is not None:
            log_run.info(message)
                
    # Properties that interface with the tracked array system.
    @property
    def all_values(self):
        """Current petrophysical values."""
        if self._tracked_state is None:
            return None
        return self._tracked_state["arr"]
    
    @property
    def orig_values(self):
        """Original petrophysical values."""
        if self._tracked_state is None:
            return None
        return self._tracked_state["orig_arr"]
    
    @property
    def petro_prev(self):
        """Previous petrophysical values."""
        if self._tracked_state is None:
            return None
        return self._tracked_state["prev_arr"]

    @property
    def temp(self):
        """Temporary values - alias for current values."""
        return self.all_values

    def init_tracked_array(self, initial_values):
        """Initialize the tracked array system with initial petrophysical values."""
        self._tracked_state = {
            "next_id": len(initial_values),
            "dtype": float,
            "items": [
                (float(initial_values[i]), i, "orig", float(initial_values[i]), float(initial_values[i]), i)  
                # (value, id, origin, orig_value, prev_value, orig_index)
                for i in range(len(initial_values))
            ],
            "last_op": None
        }
        self._resort()
        return self._tracked_state

    def _resort(self):
        """Sort items and rebuild mapping + arrays."""
        self._tracked_state["items"].sort(key=lambda x: x[0])  # sort by current value

        # Unpack sorted tuples.
        values, ids, origins, orig_vals, prev_vals, orig_inds = zip(*self._tracked_state["items"])

        # build arrays.
        self._tracked_state["arr"] = np.array(values, dtype=self._tracked_state["dtype"])          # current
        self._tracked_state["orig_arr"] = np.array(orig_vals, dtype=self._tracked_state["dtype"])  # original
        self._tracked_state["prev_arr"] = np.array(prev_vals, dtype=self._tracked_state["dtype"])  # previous

        # Build mapping by unique ID.
        self._tracked_state["mapping"] = {
            idx: {
                "ind": new_i,       # current position in sorted arr
                "val": v,           # current value
                "origin": origin,   # 'orig' or 'birth'
                "orig_v": orig_v,   # original (or first) value
                "prev_v": prev,     # last overwritten value
                "orig_ind": orig_i  # original array index (None for insertions)
            }
            for new_i, (v, idx, origin, orig_v, prev, orig_i)
            in enumerate(self._tracked_state["items"])
        }

        # Reverse lookup: orig_ind → id
        self._tracked_state["orig_lookup"] = {
            entry["orig_ind"]: idx
            for idx, entry in self._tracked_state["mapping"].items()
            if entry["orig_ind"] is not None
        }

    def insert_value(self, value):
        """Insert a new petrophysical value (e.g., from birth of a unit)."""
        if value is not None and self._tracked_state is not None: 
            new_id = self._tracked_state["next_id"]
            self._tracked_state["next_id"] += 1
            value = float(value)
            self._tracked_state["items"].append((value, new_id, "birth", value, None, None))
            self._tracked_state["last_op"] = ("insert", new_id)
            self._resort()
            return new_id
        return None

    def remove_inserted_value(self, id_remove):
        """Remove an inserted petrophysical value by ID (only if origin == 'birth')."""
        if id_remove not in self._tracked_state["mapping"]:
            raise KeyError(f"No element with ID {id_remove}")
        if self._tracked_state["mapping"][id_remove]["origin"] != "birth":
            raise ValueError(f"Element {id_remove} is not an inserted (i.e., birthed) value")

        # find removed entry.
        removed_entry = None
        new_items = []
        for tup in self._tracked_state["items"]:
            if tup[1] == id_remove:
                removed_entry = tup
            else:
                new_items.append(tup)

        self._tracked_state["items"] = new_items
        self._tracked_state["last_removed"] = removed_entry
        self._tracked_state["last_op"] = ("remove", id_remove)
        self._resort()

    def modify_value_by_index(self, ind: int, new_value: float)-> None:
        """Modify a petrophysical value by its current sorted index."""
        if ind < 0 or ind >= len(self._tracked_state["items"]):
            raise IndexError(f"Index {ind} is out of bounds")

        new_value = float(new_value)
        v, idx, origin, orig_v, prev_v, orig_i = self._tracked_state["items"][ind]

        # Replace with updated value, storing the previous value.
        self._tracked_state["items"][ind] = (new_value, idx, origin, orig_v, v, orig_i)
        self._tracked_state["last_op"] = ("modify", idx)  # keep track by unique ID for revert

        self._resort()

    def get_value_by_orig_index(self, orig_ind):
        """Return mapping entry for the element that started at orig_ind."""
        orig_ind = int(orig_ind)   # ensure it's hashable (plain int)
        if orig_ind not in self._tracked_state["orig_lookup"]:
            raise KeyError(f"No surviving element with original index {orig_ind}")
        uid = self._tracked_state["orig_lookup"][orig_ind]
        return self._tracked_state["mapping"][uid]
    
    def remove_value(self, value=None, index=None, item_id=None, allow_remove_original=False):
        """
        Remove a petrophysical value from tracking by value, index, or ID.
        More flexible than remove_inserted_value - can remove any unit (original or birthed).
        
        Parameters
        ----------
        value : float, optional
            The petrophysical value to remove (removes first match)
        index : int, optional  
            The current sorted index to remove
        item_id : int, optional
            The unique ID to remove
        allow_remove_original : bool, default=False
            If False, prevents removal of units with origin='orig'
            If True, allows removal of any unit
            
        Returns
        -------
        tuple : (removed_value, removed_id, removed_origin) or None if not found or blocked
        
        Raises
        ------
        KeyError : If item_id not found
        IndexError : If index out of bounds
        ValueError : If no parameter provided or if trying to remove protected original unit
        
        Notes
        -----
        - Only one parameter should be provided
        - Priority order: item_id > index > value
        - Stores removed entry in last_removed for potential revert
        - By default, original units are PROTECTED and cannot be removed
        - Set allow_remove_original=True to override protection (use with caution!)
        
        Examples
        --------
        >>> # Remove a birthed unit (allowed by default)
        >>> petrovals.remove_value(index=3)
        (2650.5, 3, 'birth')
        
        >>> # Try to remove original unit (blocked by default)
        >>> petrovals.remove_value(index=0)
        ValueError: Cannot remove original unit (index=0, val=2600.0). 
                    Set allow_remove_original=True to override.
        
        >>> # Force removal of original unit (if really needed)
        >>> petrovals.remove_value(index=0, allow_remove_original=True)
        (2600.0, 0, 'orig')
        """
        if not self._tracked_state:
            return None
        
        # Validate input - exactly one parameter should be provided
        params_provided = sum([value is not None, index is not None, item_id is not None])
        if params_provided == 0:
            raise ValueError("Must provide at least one of: value, index, or item_id")
        if params_provided > 1:
            import warnings
            warnings.warn("Multiple parameters provided. Using priority: item_id > index > value")
            
        # Determine which item to remove
        target_entry = None
        
        if item_id is not None:
            # Remove by unique ID (highest priority)
            if item_id not in self._tracked_state["mapping"]:
                raise KeyError(f"No element with ID {item_id}")
            for tup in self._tracked_state["items"]:
                if tup[1] == item_id:
                    target_entry = tup
                    break
                    
        elif index is not None:
            # Remove by current sorted index
            if index < 0 or index >= len(self._tracked_state["items"]):
                raise IndexError(f"Index {index} is out of bounds (0-{len(self._tracked_state['items'])-1})")
            target_entry = self._tracked_state["items"][index]
            
        elif value is not None:
            # Remove by value (first match, lowest priority)
            for tup in self._tracked_state["items"]:
                if np.isclose(tup[0], float(value)):
                    target_entry = tup
                    break
        
        if target_entry is None:
            return None
        
        # Extract info from target entry
        removed_value, removed_id, removed_origin = target_entry[0], target_entry[1], target_entry[2]
        
        # PROTECTION: Check if trying to remove an original unit
        if removed_origin == 'orig' and not allow_remove_original:
            # Get the index for better error message
            target_index = None
            for i, tup in enumerate(self._tracked_state["items"]):
                if tup[1] == removed_id:
                    target_index = i
                    break
            
            raise ValueError(
                f"Cannot remove original unit (index={target_index}, val={removed_value:.2f}, id={removed_id}). "
                f"Original units are protected by default. "
                f"Set allow_remove_original=True to override this protection."
            )
        
        # Remove the entry from items list
        self._tracked_state["items"] = [
            tup for tup in self._tracked_state["items"] 
            if tup[1] != removed_id
        ]
        
        # Store for potential revert operation
        self._tracked_state["last_removed"] = target_entry
        self._tracked_state["last_op"] = ("remove", removed_id)
        
        # Rebuild sorted arrays and mappings
        self._resort()
        
        return (removed_value, removed_id, removed_origin)

    def revert_last_change(self):
        """Revert the most recent modification, insertion, or removal."""
        if self._tracked_state["last_op"] is None:
            return

        op_type, target_id = self._tracked_state["last_op"]

        if op_type == "insert":
            # remove the just-inserted element
            self._tracked_state["items"] = [
                (v, idx, origin, orig_v, prev_v, orig_i)
                for (v, idx, origin, orig_v, prev_v, orig_i) in self._tracked_state["items"]
                if idx != target_id
            ]

        elif op_type == "modify":
            # restore previous value
            for i, (v, idx, origin, orig_v, prev_v, orig_i) in enumerate(self._tracked_state["items"]):
                if idx == target_id and prev_v is not None:
                    self._tracked_state["items"][i] = (prev_v, idx, origin, orig_v, prev_v, orig_i)
                    break

        elif op_type == "remove":
            # put the removed element back
            removed_entry = self._tracked_state.get("last_removed")
            if removed_entry is not None and removed_entry[1] == target_id:
                self._tracked_state["items"].append(removed_entry)

        self._tracked_state["last_op"] = None
        self._tracked_state.pop("last_removed", None)
        self._resort()

    def set_prev_as_current(self):
        """Set prev_v for all items equal to their current val."""
        self._tracked_state["items"] = [
            (v, idx, origin, orig_v, v, orig_i)
            for (v, idx, origin, orig_v, prev_v, orig_i) in self._tracked_state["items"]
        ]
        self._resort()

    def diff_from_orig(self):
        """Calculate the sum of squares of differences between current and original petro values."""
        sum_sq_diff_curr = 0
        sum_sq_diff_prev = 0
        for _, entry in self._tracked_state["mapping"].items():
            if entry["origin"] in ("birth", "death"):
                continue  # skip inserted/removed elements
            cur = entry["val"]
            orig = entry["orig_v"]
            prev = entry["prev_v"]
            sum_sq_diff_curr += (cur - orig) * (cur - orig)
            sum_sq_diff_prev += (prev - orig) * (prev - orig)
            
        return sum_sq_diff_curr, sum_sq_diff_prev
    
    def calc_petro_pert(self, model_with_pert) -> np.ndarray:
        """Calculate perturbation from model without modifying state."""
        unique_vals_pert = np.unique(model_with_pert[model_with_pert != 0.])
        if len(unique_vals_pert) != 1:
            raise qc.ModelPerturbationError("Perturbation should have only one non-zero value!")
        return np.array([0, unique_vals_pert[0]]) 

    def set_pert_from_model(self, model_with_pert) -> None:
        """Set perturbation from geological model."""
        self.pert = self.calc_petro_pert(model_with_pert)
    
    def resync_with_model(self, model_curr):
        """
        Resynchronize the tracked array with current model values.
        Use this after operations like birth that may change the model
        outside of the tracked array system.
        """
        current_model_values = tu.get_petro_values(model_curr)

        existing_values = set(self.all_values)
        current_values = set(current_model_values)

        # Handle births: insert new value. 
        new_values = current_values - existing_values
        for new_val in new_values:
            self.insert_value(new_val)

        # Handle deaths: remove values.
        removed_values = existing_values - current_values
        for old_val in removed_values:
            self.remove_value(value=old_val, allow_remove_original=False)   

    def update_prev(self, accepted):
        """Update previous values - now handled by tracked array system."""
        if accepted and self._tracked_state is not None:
            self.set_prev_as_current()

    def revert_to_prev(self):
        """Revert to previous values using tracked array system."""
        if self._tracked_state is not None:
            self.revert_last_change()

    # Access to internal state for compatibility
    @property
    def tracked_state(self):
        """Access to the internal tracked state for advanced operations."""
        return self._tracked_state
    
    def has_birthed_units(self):
        """
        Check if any units resulted from birth operations.
        
        Returns
        -------
        bool
            True if at least one birthed unit exists, False if all units are original
            
        Examples
        --------
        >>> # Check before attempting death
        >>> if not petrovals.has_birthed_units():
        >>>     log_run.info("No birthed units - skipping death")
        >>>     return
        
        >>> # Conditional logic
        >>> if petrovals.has_birthed_units():
        >>>     # Can propose death
        >>> else:
        >>>     # All original - protect them
        """
        if self._tracked_state is None:
            return False
    
        return any(item[2] == 'birth' for item in self._tracked_state["items"])
