"""File-based task queue for NFS-safe distributed experiment execution.

Uses atomic rename operations for task claiming, which is reliable on NFS
unlike SQLite's locking mechanisms.

Directory structure:
    queue_dir/
        pending/     - Tasks waiting to be claimed
        running/     - Tasks currently being processed
        completed/   - Successfully finished tasks
        failed/      - Tasks that failed with errors
"""

import json
import os
import socket
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Experiment:
    """Represents a single experiment configuration."""
    id: int
    config_hash: str
    config: dict
    seed: int
    status: ExperimentStatus
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    best_model_path: Optional[str] = None
    wandb_run_id: Optional[str] = None


@dataclass
class Evaluation:
    """Represents evaluation metrics at a specific epoch."""
    id: int
    experiment_id: int
    epoch: int
    metrics: dict
    created_at: datetime


def compute_config_hash(config: dict) -> str:
    """Compute a deterministic hash for a configuration dict."""
    def sort_dict(d):
        if isinstance(d, dict):
            return {k: sort_dict(v) for k, v in sorted(d.items())}
        elif isinstance(d, list):
            return [sort_dict(x) for x in d]
        return d
    
    sorted_config = sort_dict(config)
    config_str = json.dumps(sorted_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class FileTaskQueue:
    """
    File-based task queue for distributed experiment execution.
    
    Uses atomic rename operations for task claiming, which works reliably
    on NFS unlike SQLite's locking mechanisms.
    
    Each experiment is stored as a JSON file. Workers claim tasks by
    atomically renaming files from pending/ to running/.
    """
    
    SUBDIRS = ["pending", "running", "completed", "failed"]
    
    def __init__(self, queue_dir: str | Path):
        """
        Initialize the file task queue.
        
        Args:
            queue_dir: Root directory for the task queue.
        """
        self.queue_dir = Path(queue_dir)
        self._init_dirs()
        self._id_counter_file = self.queue_dir / ".id_counter"
    
    def _init_dirs(self):
        """Create queue subdirectories if they don't exist."""
        for subdir in self.SUBDIRS:
            (self.queue_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _get_next_id(self) -> int:
        """Get the next experiment ID (thread-safe via atomic file operations)."""
        # Use a counter file to track IDs
        counter_file = self._id_counter_file
        
        # Try to atomically increment the counter
        for _ in range(100):  # Retry loop for concurrent access
            try:
                if counter_file.exists():
                    current = int(counter_file.read_text().strip())
                else:
                    current = 0
                
                next_id = current + 1
                
                # Write to temp file and rename (atomic)
                temp_file = counter_file.with_suffix('.tmp')
                temp_file.write_text(str(next_id))
                os.rename(temp_file, counter_file)
                
                return next_id
            except (OSError, ValueError):
                # Concurrent modification, retry
                continue
        
        raise RuntimeError("Failed to allocate experiment ID after 100 retries")
    
    def _task_filename(self, exp_id: int, seed: int, config_hash: str) -> str:
        """Generate the filename for a task."""
        return f"exp_{exp_id:06d}_seed_{seed}_{config_hash}.json"
    
    def _exists_by_hash_and_seed(self, config_hash: str, seed: int) -> bool:
        """Check if an experiment with this config_hash and seed already exists."""
        # Filename pattern: exp_XXXXXX_seed_Y_HASH.json
        pattern = f"exp_*_seed_{seed}_{config_hash}.json"
        for subdir in self.SUBDIRS:
            dir_path = self.queue_dir / subdir
            if list(dir_path.glob(pattern)):
                return True
        return False
    
    def _find_task_file(self, exp_id: int) -> Optional[tuple[Path, dict]]:
        """Find a task file by experiment ID across all directories."""
        for subdir in self.SUBDIRS:
            dir_path = self.queue_dir / subdir
            for task_file in dir_path.glob(f"exp_{exp_id:06d}_*.json"):
                try:
                    data = json.loads(task_file.read_text())
                    return task_file, data
                except (json.JSONDecodeError, OSError):
                    continue
        return None
    
    def _read_task(self, task_file: Path) -> Optional[dict]:
        """Read and parse a task file."""
        try:
            return json.loads(task_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    
    def _write_task(self, task_file: Path, data: dict):
        """Write task data to file (atomic via temp file + rename)."""
        temp_file = task_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2, default=str))
        os.rename(temp_file, task_file)
    
    def insert_experiment(
        self,
        config: dict,
        seed: int,
        skip_existing: bool = True
    ) -> Optional[int]:
        """
        Insert a new experiment configuration.
        
        Args:
            config: Experiment configuration dictionary.
            seed: Random seed for this run.
            skip_existing: If True, skip if (config_hash, seed) already exists.
            
        Returns:
            The experiment ID, or None if skipped.
        """
        config_hash = compute_config_hash(config)
        
        if skip_existing:
            # Fast check using filename pattern (no file reading needed)
            if self._exists_by_hash_and_seed(config_hash, seed):
                return None
        
        exp_id = self._get_next_id()
        filename = self._task_filename(exp_id, seed, config_hash)
        
        task_data = {
            "id": exp_id,
            "config_hash": config_hash,
            "config": config,
            "seed": seed,
            "status": ExperimentStatus.PENDING.value,
            "worker_id": None,
            "started_at": None,
            "completed_at": None,
            "error_message": None,
            "best_model_path": None,
            "wandb_run_id": None,
            "created_at": datetime.now().isoformat(),
            "evaluations": []
        }
        
        task_file = self.queue_dir / "pending" / filename
        self._write_task(task_file, task_data)
        
        return exp_id
    
    def claim_pending_task(self, worker_id: Optional[str] = None) -> Optional[Experiment]:
        """
        Atomically claim a pending task for processing.
        
        Uses atomic rename to prevent race conditions between workers.
        Tasks are claimed in order by seed first, then randomly within each seed.
        
        Args:
            worker_id: Identifier for the worker claiming the task.
                       Defaults to hostname-pid.
                       
        Returns:
            The claimed Experiment, or None if no pending tasks.
        """
        if worker_id is None:
            worker_id = f"{socket.gethostname()}-{os.getpid()}"
        
        pending_dir = self.queue_dir / "pending"
        running_dir = self.queue_dir / "running"
        
        # Get all pending tasks and sort by seed, then shuffle within each seed
        pending_files = list(pending_dir.glob("exp_*.json"))
        if not pending_files:
            return None
        
        # Parse tasks to get seed info
        tasks_with_seed = []
        for task_file in pending_files:
            data = self._read_task(task_file)
            if data:
                tasks_with_seed.append((task_file, data, data.get("seed", 0)))
        
        if not tasks_with_seed:
            return None
        
        # Sort by seed, then shuffle within each seed group for random selection
        tasks_with_seed.sort(key=lambda x: x[2])
        
        # Group by seed and shuffle each group
        seed_groups = {}
        for task_file, data, seed in tasks_with_seed:
            if seed not in seed_groups:
                seed_groups[seed] = []
            seed_groups[seed].append((task_file, data))
        
        # Flatten back, shuffled within each seed
        ordered_tasks = []
        for seed in sorted(seed_groups.keys()):
            group = seed_groups[seed]
            random.shuffle(group)
            ordered_tasks.extend(group)
        
        # Try to claim a task (may fail if another worker claims it first)
        for task_file, data in ordered_tasks:
            new_path = running_dir / task_file.name
            
            try:
                # Atomic rename - this is the key to preventing race conditions
                os.rename(task_file, new_path)
                
                # Successfully claimed! Update the task data
                now = datetime.now().isoformat()
                data["status"] = ExperimentStatus.RUNNING.value
                data["worker_id"] = worker_id
                data["started_at"] = now
                
                self._write_task(new_path, data)
                
                return Experiment(
                    id=data["id"],
                    config_hash=data["config_hash"],
                    config=data["config"],
                    seed=data["seed"],
                    status=ExperimentStatus.RUNNING,
                    worker_id=worker_id,
                    started_at=datetime.fromisoformat(now)
                )
            except OSError:
                # Another worker claimed it first, try next
                continue
        
        return None
    
    def mark_completed(
        self,
        experiment_id: int,
        best_model_path: Optional[str] = None,
        best_policy_path: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        final_metrics: Optional[dict] = None,
        best_epoch: Optional[int] = None
    ):
        """Mark an experiment as completed."""
        running_dir = self.queue_dir / "running"
        completed_dir = self.queue_dir / "completed"
        
        # Find the task file
        for task_file in running_dir.glob(f"exp_{experiment_id:06d}_*.json"):
            data = self._read_task(task_file)
            if data:
                data["status"] = ExperimentStatus.COMPLETED.value
                data["completed_at"] = datetime.now().isoformat()
                data["best_model_path"] = best_model_path
                data["best_policy_path"] = best_policy_path
                data["wandb_run_id"] = wandb_run_id
                data["best_epoch"] = best_epoch
                
                # Store final metrics (regret, mean_rew, etc.)
                if final_metrics:
                    # Clean up metric keys (remove eval/ prefix) and filter numeric values
                    cleaned_metrics = {}
                    for key, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            clean_key = key.replace("eval/", "")
                            cleaned_metrics[clean_key] = value
                    data["final_metrics"] = cleaned_metrics
                
                # Write updated data
                self._write_task(task_file, data)
                
                # Move to completed
                new_path = completed_dir / task_file.name
                os.rename(task_file, new_path)
                return
    
    def mark_failed(self, experiment_id: int, error_message: str):
        """Mark an experiment as failed with an error message."""
        running_dir = self.queue_dir / "running"
        failed_dir = self.queue_dir / "failed"
        
        # Find the task file
        for task_file in running_dir.glob(f"exp_{experiment_id:06d}_*.json"):
            data = self._read_task(task_file)
            if data:
                data["status"] = ExperimentStatus.FAILED.value
                data["completed_at"] = datetime.now().isoformat()
                data["error_message"] = error_message
                
                # Write updated data
                self._write_task(task_file, data)
                
                # Move to failed
                new_path = failed_dir / task_file.name
                os.rename(task_file, new_path)
                return
    
    def log_evaluation(
        self,
        experiment_id: int,
        epoch: int,
        metrics: dict
    ):
        """
        Log evaluation metrics for an experiment at a specific epoch.
        
        Appends to the evaluations list in the task file.
        """
        # Find the task file (should be in running)
        running_dir = self.queue_dir / "running"
        
        for task_file in running_dir.glob(f"exp_{experiment_id:06d}_*.json"):
            data = self._read_task(task_file)
            if data:
                # Check if this epoch already exists, update if so
                evaluations = data.get("evaluations", [])
                found = False
                for eval_entry in evaluations:
                    if eval_entry.get("epoch") == epoch:
                        eval_entry["metrics"] = metrics
                        eval_entry["created_at"] = datetime.now().isoformat()
                        found = True
                        break
                
                if not found:
                    evaluations.append({
                        "epoch": epoch,
                        "metrics": metrics,
                        "created_at": datetime.now().isoformat()
                    })
                
                data["evaluations"] = evaluations
                self._write_task(task_file, data)
                return
    
    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get an experiment by ID."""
        result = self._find_task_file(experiment_id)
        if result is None:
            return None
        
        task_file, data = result
        
        return Experiment(
            id=data["id"],
            config_hash=data["config_hash"],
            config=data["config"],
            seed=data["seed"],
            status=ExperimentStatus(data["status"]),
            worker_id=data.get("worker_id"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            best_model_path=data.get("best_model_path"),
            wandb_run_id=data.get("wandb_run_id")
        )
    
    def get_evaluations(self, experiment_id: int) -> list[Evaluation]:
        """Get all evaluations for an experiment, ordered by epoch."""
        result = self._find_task_file(experiment_id)
        if result is None:
            return []
        
        task_file, data = result
        evaluations = data.get("evaluations", [])
        
        return [
            Evaluation(
                id=i,
                experiment_id=experiment_id,
                epoch=e["epoch"],
                metrics=e["metrics"],
                created_at=datetime.fromisoformat(e["created_at"])
            )
            for i, e in enumerate(sorted(evaluations, key=lambda x: x["epoch"]))
        ]
    
    def get_status_summary(self) -> dict[str, int]:
        """Get counts of experiments by status."""
        summary = {s.value: 0 for s in ExperimentStatus}
        
        for status in ExperimentStatus:
            dir_path = self.queue_dir / status.value
            if dir_path.exists():
                summary[status.value] = len(list(dir_path.glob("exp_*.json")))
        
        summary["total"] = sum(summary.values())
        return summary
    
    def reset_failed_to_pending(self) -> int:
        """Reset all failed experiments to pending. Returns count reset."""
        failed_dir = self.queue_dir / "failed"
        pending_dir = self.queue_dir / "pending"
        
        count = 0
        for task_file in failed_dir.glob("exp_*.json"):
            data = self._read_task(task_file)
            if data:
                data["status"] = ExperimentStatus.PENDING.value
                data["worker_id"] = None
                data["started_at"] = None
                data["completed_at"] = None
                data["error_message"] = None
                
                self._write_task(task_file, data)
                
                new_path = pending_dir / task_file.name
                os.rename(task_file, new_path)
                count += 1
        
        return count
    
    def reset_running_to_pending(self) -> int:
        """
        Reset all running experiments to pending.
        Useful for recovering from interrupted runs.
        Returns count reset.
        """
        running_dir = self.queue_dir / "running"
        pending_dir = self.queue_dir / "pending"
        
        count = 0
        for task_file in running_dir.glob("exp_*.json"):
            data = self._read_task(task_file)
            if data:
                data["status"] = ExperimentStatus.PENDING.value
                data["worker_id"] = None
                data["started_at"] = None
                
                self._write_task(task_file, data)
                
                new_path = pending_dir / task_file.name
                os.rename(task_file, new_path)
                count += 1
        
        return count
    
    def get_all_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> list[Experiment]:
        """Get all experiments, optionally filtered by status."""
        experiments = []
        
        dirs_to_check = [status.value] if status else self.SUBDIRS
        
        for subdir in dirs_to_check:
            dir_path = self.queue_dir / subdir
            if not dir_path.exists():
                continue
            
            for task_file in dir_path.glob("exp_*.json"):
                data = self._read_task(task_file)
                if data:
                    experiments.append(Experiment(
                        id=data["id"],
                        config_hash=data["config_hash"],
                        config=data["config"],
                        seed=data["seed"],
                        status=ExperimentStatus(data["status"]),
                        worker_id=data.get("worker_id"),
                        started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                        error_message=data.get("error_message"),
                        best_model_path=data.get("best_model_path"),
                        wandb_run_id=data.get("wandb_run_id")
                    ))
        
        # Sort by ID
        experiments.sort(key=lambda x: x.id)
        return experiments
    
    def export_results(self) -> list[dict]:
        """
        Export all completed experiments with their evaluations.
        
        Returns a list of dicts suitable for conversion to DataFrame/JSON.
        """
        experiments = self.get_all_experiments(ExperimentStatus.COMPLETED)
        results = []
        
        for exp in experiments:
            evaluations = self.get_evaluations(exp.id)
            
            # Flatten config into result
            result = {
                "experiment_id": exp.id,
                "config_hash": exp.config_hash,
                "seed": exp.seed,
                "best_model_path": exp.best_model_path,
                "wandb_run_id": exp.wandb_run_id,
                "started_at": exp.started_at.isoformat() if exp.started_at else None,
                "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
                **{f"config.{k}": v for k, v in exp.config.items()},
            }
            
            # Add final evaluation metrics
            if evaluations:
                final_eval = evaluations[-1]
                result["final_epoch"] = final_eval.epoch
                for k, v in final_eval.metrics.items():
                    result[f"final.{k}"] = v
            
            results.append(result)
        
        return results
    
    def delete_experiments_by_filter(
        self,
        config_filter: dict[str, any],
        status: Optional[ExperimentStatus] = None,
        dry_run: bool = False,
    ) -> tuple[int, list[dict]]:
        """
        Delete experiments matching config filters.
        
        Args:
            config_filter: Dictionary mapping config keys to values to match.
                           Keys should be config field names (without "config." prefix).
            status: If provided, only consider experiments with this status.
            dry_run: If True, don't actually delete files, just return what would be deleted.
            
        Returns:
            Tuple of (count of matched experiments, list of matched experiment summaries)
        """
        dirs_to_check = [status.value] if status else self.SUBDIRS
        matched = []
        
        for subdir in dirs_to_check:
            dir_path = self.queue_dir / subdir
            if not dir_path.exists():
                continue
            
            for task_file in list(dir_path.glob("exp_*.json")):
                data = self._read_task(task_file)
                if data is None:
                    continue
                
                config = data.get("config", {})
                
                # Check if all filters match
                matches = True
                for key, value in config_filter.items():
                    config_value = config.get(key)
                    
                    # Handle type conversion for comparison
                    if config_value is None:
                        matches = False
                        break
                    
                    # Convert filter value to match config value type
                    try:
                        if isinstance(config_value, bool):
                            if isinstance(value, str):
                                typed_value = value.lower() in ("true", "1", "yes")
                            else:
                                typed_value = bool(value)
                        elif isinstance(config_value, int):
                            typed_value = int(value)
                        elif isinstance(config_value, float):
                            typed_value = float(value)
                        else:
                            typed_value = str(value)
                        
                        if config_value != typed_value:
                            matches = False
                            break
                    except (ValueError, TypeError):
                        # If conversion fails, try string comparison
                        if str(config_value) != str(value):
                            matches = False
                            break
                
                if matches:
                    summary = {
                        "id": data.get("id"),
                        "seed": data.get("seed"),
                        "status": data.get("status"),
                        "config_hash": data.get("config_hash"),
                        "file_path": str(task_file),
                    }
                    # Add filtered config values for display
                    for key in config_filter.keys():
                        summary[f"config.{key}"] = config.get(key)
                    
                    matched.append(summary)
                    
                    if not dry_run:
                        task_file.unlink()
        
        return len(matched), matched
    
    def delete_experiments_not_in_hashes(
        self,
        valid_hashes: set[str],
        status: Optional[ExperimentStatus] = None,
        dry_run: bool = False,
    ) -> tuple[int, list[dict]]:
        """
        Delete experiments whose config_hash is NOT in the provided set.
        
        This is useful for purging experiments that don't match a given grid.
        
        Args:
            valid_hashes: Set of valid config hashes to keep.
            status: If provided, only consider experiments with this status.
            dry_run: If True, don't actually delete files, just return what would be deleted.
            
        Returns:
            Tuple of (count of matched experiments, list of matched experiment summaries)
        """
        dirs_to_check = [status.value] if status else self.SUBDIRS
        matched = []
        
        for subdir in dirs_to_check:
            dir_path = self.queue_dir / subdir
            if not dir_path.exists():
                continue
            
            for task_file in list(dir_path.glob("exp_*.json")):
                data = self._read_task(task_file)
                if data is None:
                    continue
                
                config_hash = data.get("config_hash")
                
                # If this experiment's hash is NOT in valid_hashes, mark for deletion
                if config_hash not in valid_hashes:
                    config = data.get("config", {})
                    summary = {
                        "id": data.get("id"),
                        "seed": data.get("seed"),
                        "status": data.get("status"),
                        "config_hash": config_hash,
                        "file_path": str(task_file),
                        # Include some identifying config values
                        "config.env_id": config.get("env_id"),
                    }
                    
                    matched.append(summary)
                    
                    if not dry_run:
                        task_file.unlink()
        
        return len(matched), matched

