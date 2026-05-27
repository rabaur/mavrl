"""Experiment configuration grid with support for conditional parameters."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
from tqdm import tqdm

from mavrl_experiments.file_queue import FileTaskQueue, compute_config_hash


@dataclass
class Parameter:
    """A parameter with its possible values."""
    name: str
    values: list[Any]
    condition: Optional[Callable[[dict], bool]] = None
    
    def applies_to(self, config: dict) -> bool:
        """Check if this parameter should be included in the given config."""
        if self.condition is None:
            return True
        return self.condition(config)


class ExperimentGrid:
    """
    Defines a grid of experiment configurations with support for:
    - Simple parameter sweeps
    - Conditional parameters (only apply when condition is met)
    - Cross-parameter validation (filter invalid configs)
    - Explicit seed specification via grid.add("seed", [...])
    - Deduplication via config hashing
    """
    
    def __init__(
        self, 
        base_config: Optional[dict] = None,
        all_params: Optional[set[str]] = None
    ):
        """
        Initialize an experiment grid.
        
        Args:
            base_config: Default values for all configurations.
            all_params: Optional set of all valid parameter names. If provided,
                       the summary will highlight which parameters are not 
                       specified by the grid (using defaults from train.py).
                       Can be obtained via: set(vars(get_default_args()).keys())
        """
        self.base_config = base_config or {}
        self.parameters: list[Parameter] = []
        self.validators: list[Callable[[dict], bool]] = []
        self.all_params = all_params
    
    def add(self, name: str, values: list[Any]) -> "ExperimentGrid":
        """
        Add a parameter to sweep over.
        
        Args:
            name: Parameter name (should match train.py argument names).
            values: List of values to try.
            
        Returns:
            self for chaining.
        """
        self.parameters.append(Parameter(name=name, values=values))
        return self
    
    def add_conditional(
        self, 
        name: str, 
        values: list[Any],
        condition: Callable[[dict], bool]
    ) -> "ExperimentGrid":
        """
        Add a parameter that only applies when a condition is met.
        
        Args:
            name: Parameter name.
            values: List of values to try.
            condition: Function that takes partial config dict and returns
                      True if this parameter should be included.
                      
        Returns:
            self for chaining.
        """
        self.parameters.append(Parameter(name=name, values=values, condition=condition))
        return self
    
    def add_validator(self, validator: Callable[[dict], bool]) -> "ExperimentGrid":
        """
        Add a validator function that checks if a complete config is valid.
        
        Validators are called on complete configs after all parameters are set.
        They should return True if the config is valid, False otherwise.
        
        Args:
            validator: Function that takes a complete config dict and returns
                      True if valid, False if invalid.
                      
        Returns:
            self for chaining.
            
        Example:
            # Ensure at least one of n_pref_samples or n_demo_samples > 0
            grid.add_validator(
                lambda c: c.get("n_pref_samples", 0) > 0 or c.get("n_demo_samples", 0) > 0
            )
        """
        self.validators.append(validator)
        return self
    
    def get_specified_params(self) -> set[str]:
        """
        Get all parameter names that are specified by this grid.
        
        Returns:
            Set of parameter names from base_config and added parameters.
        """
        specified = set(self.base_config.keys())
        for param in self.parameters:
            specified.add(param.name)
        return specified
    
    def get_unspecified_params(self) -> set[str]:
        """
        Get parameter names that exist in all_params but are not specified by this grid.
        
        Returns:
            Set of unspecified parameter names. Empty if all_params was not provided.
        """
        if self.all_params is None:
            return set()
        return self.all_params - self.get_specified_params()
    
    def get_invalid_params(self) -> set[str]:
        """
        Get parameter names that are specified by this grid but don't exist in all_params.
        
        These are likely typos or outdated parameter names that would be silently ignored.
        
        Note: Parameters prefixed with 'env_params.' are always allowed as they
        represent environment-specific settings.
        
        Returns:
            Set of invalid parameter names. Empty if all_params was not provided.
        """
        if self.all_params is None:
            return set()
        specified = self.get_specified_params()
        # Filter out env_params.* parameters as they're always valid
        non_env_params = {p for p in specified if not p.startswith("env_params.")}
        return non_env_params - self.all_params
    
    def validate_params(self) -> None:
        """
        Validate that all specified parameters are valid train.py arguments.
        
        Raises:
            ValueError: If any parameters are specified that don't exist in all_params.
                       This catches typos like "use_imitiation_learning" instead of 
                       "use_imitation_learning".
        """
        invalid = self.get_invalid_params()
        if invalid:
            # Try to suggest corrections for typos
            suggestions = []
            for inv_param in sorted(invalid):
                # Find similar valid params (simple edit distance heuristic)
                similar = self._find_similar_params(inv_param)
                if similar:
                    suggestions.append(f"  - '{inv_param}' (did you mean '{similar}'?)")
                else:
                    suggestions.append(f"  - '{inv_param}'")
            
            raise ValueError(
                f"Invalid parameter names in experiment grid (not recognized by train.py):\n"
                + "\n".join(suggestions) + "\n\n"
                f"These parameters will be silently ignored, likely causing unexpected behavior.\n"
                f"Please check for typos or use get_all_train_params() to see valid parameter names."
            )
    
    def _find_similar_params(self, invalid_param: str, max_distance: int = 3) -> Optional[str]:
        """Find a similar valid parameter name (for typo suggestions)."""
        if self.all_params is None:
            return None
        
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Compute Levenshtein distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            
            return prev_row[-1]
        
        best_match = None
        best_distance = max_distance + 1
        
        for valid_param in self.all_params:
            dist = levenshtein_distance(invalid_param.lower(), valid_param.lower())
            if dist < best_distance:
                best_distance = dist
                best_match = valid_param
        
        return best_match if best_distance <= max_distance else None
    
    def _generate_configs_recursive(
        self,
        param_idx: int,
        current_config: dict
    ) -> list[dict]:
        """Recursively generate all valid configurations."""
        if param_idx >= len(self.parameters):
            return [current_config.copy()]
        
        param = self.parameters[param_idx]
        
        # Check if this parameter applies to current config
        if not param.applies_to(current_config):
            # Skip this parameter, continue with next
            return self._generate_configs_recursive(param_idx + 1, current_config)
        
        # Generate configs for each value of this parameter
        configs = []
        for value in param.values:
            new_config = current_config.copy()
            # Safety check: don't overwrite existing values (catches overlapping conditionals)
            if param.name in new_config:
                raise ValueError(
                    f"Parameter '{param.name}' already set to {new_config[param.name]!r}, "
                    f"cannot overwrite with {value!r}. Check for overlapping conditions."
                )
            new_config[param.name] = value
            configs.extend(self._generate_configs_recursive(param_idx + 1, new_config))
        
        return configs
    
    def generate_configs(self) -> list[tuple[dict, int]]:
        """
        Generate all valid (config, seed) combinations.

        Requires that ``seed`` has been added as a grid parameter via
        ``grid.add("seed", [0, 1, ...])``.  The seed value is extracted
        from each generated config so the returned tuples have the same
        ``(config_dict, seed)`` shape the rest of the pipeline expects.
        
        Returns:
            List of (config_dict, seed) tuples.
            
        Raises:
            ValueError: If ``seed`` is not a grid parameter, or if any
                parameters are invalid (not recognized by train.py).
        """
        self.validate_params()

        if not any(p.name == "seed" for p in self.parameters):
            raise ValueError(
                "Grid must include seed as a parameter: grid.add('seed', [0, 1, ...])"
            )
        
        base = self.base_config.copy()
        configs = self._generate_configs_recursive(0, base)
        
        if self.validators:
            configs = [c for c in configs if all(v(c) for v in self.validators)]
        
        # Pop seed from each config and deduplicate on (config_hash, seed).
        # Config hash deliberately excludes seed so it stays consistent with
        # FileTaskQueue._exists_by_hash_and_seed.
        results = []
        seen = set()
        for config in configs:
            seed = config.pop("seed")
            h = compute_config_hash(config)
            key = (h, seed)
            if key not in seen:
                seen.add(key)
                results.append((config, seed))
        return results
    
    def populate_db(
        self, 
        queue: FileTaskQueue,
        dry_run: bool = False
    ) -> dict[str, int]:
        """
        Populate the queue with experiment configurations.
        
        Args:
            queue: The task queue to populate (FileTaskQueue).
            dry_run: If True, don't actually insert, just return counts.
            
        Returns:
            Dict with counts: {"total": N, "inserted": M, "skipped": K}
        """
        configs = self.generate_configs()
        
        inserted = 0
        skipped = 0
        
        for config, seed in tqdm(configs, desc="Populating queue"):
            if dry_run:
                # In dry run, assume all would be inserted (no queue check)
                inserted += 1
            else:
                exp_id = queue.insert_experiment(config, seed, skip_existing=True)
                if exp_id is not None:
                    inserted += 1
                else:
                    skipped += 1
        
        return {
            "total": len(configs),
            "inserted": inserted,
            "skipped": skipped
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary of the grid."""
        configs = self.generate_configs()
        
        seed_values = sorted(set(seed for _, seed in configs))
        n_unique = len(set(compute_config_hash(c) for c, _ in configs))
        
        lines = [
            f"Experiment Grid Summary",
            f"=" * 40,
            f"Unique configurations: {n_unique}",
            f"Seeds: {seed_values}",
            f"Total experiments: {len(configs)}",
            f"",
            f"Parameters:",
        ]
        
        for param in self.parameters:
            if param.name == "seed":
                continue
            condition_str = " (conditional)" if param.condition else ""
            lines.append(f"  {param.name}: {param.values}{condition_str}")
        
        if self.base_config:
            lines.append("")
            lines.append("Base config:")
            for k, v in sorted(self.base_config.items()):
                lines.append(f"  {k}: {v}")
        
        # Show invalid parameters (typos, etc.) - this is an error condition
        invalid = self.get_invalid_params()
        if invalid:
            lines.append("")
            lines.append(f"❌ INVALID parameters ({len(invalid)}) - not recognized by train.py:")
            for name in sorted(invalid):
                similar = self._find_similar_params(name)
                if similar:
                    lines.append(f"  {name}  (did you mean '{similar}'?)")
                else:
                    lines.append(f"  {name}")
            lines.append("")
            lines.append("⚠️  These parameters will be silently ignored! Fix typos before running.")
        
        # Show unspecified parameters if all_params was provided
        unspecified = self.get_unspecified_params()
        if unspecified:
            lines.append("")
            lines.append(f"⚠️  Unspecified parameters ({len(unspecified)}) - using train.py defaults:")
            for name in sorted(unspecified):
                lines.append(f"  {name}")
        elif self.all_params is not None and not invalid:
            lines.append("")
            lines.append("✓ All parameters specified")
        
        return "\n".join(lines)


def get_all_train_params() -> set[str]:
    """
    Get all parameter names from train.py's argument parser.
    
    Returns:
        Set of all valid parameter names.
        
    Example:
        grid = ExperimentGrid(all_params=get_all_train_params())
        grid.add("env_id", ["CartPole-v1"])
        print(grid.summary())  # Will show unspecified params
    """
    from train import get_default_args
    return set(vars(get_default_args()).keys())


def get_all_transfer_params() -> set[str]:
    """
    Get all parameter names for transfer experiments from transfer.py's argument parser.
    
    Note: Parameters prefixed with 'env_params.' are always allowed as they
    represent environment-specific settings that vary by environment.
    
    Returns:
        Set of all valid transfer parameter names.
        
    Example:
        grid = ExperimentGrid(all_params=get_all_transfer_params())
        grid.add("env_id", ["LunarLander-v3"])
        grid.add("env_params.wind_power", [5.0, 10.0])  # env_params.* always allowed
        print(grid.summary())
    """
    # These are the transfer.py arguments (from transfer.py's main())
    transfer_params = {
        "env_id",
        "reward_domain",
        "env_params",
        "fb_model_path",
        "mode",
        "hidden_sizes",
        "optimal_policy_path",
        "num_samples",
        "max_num_steps",
        "gamma",
        "seed",
        "retrain_verbose",
        "no_progress_bar",
        "retrain_reward_thresh",
        "act_transform",
        "obs_transform",
        "log_wandb",
        "wandb_project",
        "wandb_entity",
        "wandb_name",
        "use_avg_state_regret",
        # Additional params used in evaluation grids
        "feedback_combo",  # Used to organize/filter model paths
        "encoder_hidden_sizes",  # Model architecture
        "checkpoint_paths",  # Multiple checkpoints for average mode
        "eval_condition",  # Evaluation condition (e.g. mavrl_pdrs, average, ground_truth)
    }
    
    return transfer_params


def make_grid_from_dict(spec: dict) -> ExperimentGrid:
    """
    Create an ExperimentGrid from a dictionary specification.
    
    Useful for loading grid definitions from YAML/JSON.
    
    Args:
        spec: Dict with keys:
            - "base": dict of base configuration values
            - "sweep": dict of param_name -> list of values
            
    Returns:
        An ExperimentGrid instance.
        
    Example:
        spec = {
            "base": {"num_epochs": 1000, "batch_size": 128},
            "sweep": {
                "env_id": ["CartPole-v1", "LunarLander-v3"],
                "lr": [1e-3, 1e-4]
            }
        }
        grid = make_grid_from_dict(spec)
    """
    grid = ExperimentGrid(base_config=spec.get("base", {}))
    
    for name, values in spec.get("sweep", {}).items():
        if not isinstance(values, list):
            values = [values]
        grid.add(name, values)
    
    return grid

