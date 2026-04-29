"""
Task scheduler for BER simulation cluster.
Assigns tasks based on worker performance and historical results.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


class WorkerPerformanceTracker:
    """Tracks worker speed/performance metrics."""

    def __init__(self, window_size=10):
        """
        Initialize tracker.

        Args:
            window_size: Number of recent FPS measurements to consider
        """
        self.window_size = window_size
        self.fps_history = defaultdict(lambda: deque(maxlen=window_size))
        self.worker_ranks = {}  # Cached ranking

    def update_fps(self, worker_id: str, fps: float):
        """Record FPS measurement for a worker."""
        if fps > 0:
            self.fps_history[worker_id].append(fps)
            self.worker_ranks = {}  # Invalidate cache

    def get_average_fps(self, worker_id: str) -> float:
        """Get average FPS for worker, or 0 if unknown."""
        if not self.fps_history[worker_id]:
            return 0.0
        return float(np.mean(list(self.fps_history[worker_id])))

    def get_worker_rank(self, worker_id: str) -> int:
        """
        Get worker rank (0=fastest, increasing=slower).
        Higher rank = slower worker.
        """
        if not self.worker_ranks:
            self._compute_ranks()
        return self.worker_ranks.get(worker_id, 999)

    def _compute_ranks(self):
        """Compute and cache worker performance ranks."""
        fps_dict = {wid: self.get_average_fps(wid) for wid in self.fps_history.keys()}

        # Sort by FPS descending (fastest first)
        sorted_workers = sorted(fps_dict.items(), key=lambda x: x[1], reverse=True)
        self.worker_ranks = {wid: rank for rank, (wid, _) in enumerate(sorted_workers)}

    def get_active_workers(self) -> List[str]:
        """Get list of workers we've seen."""
        return list(self.fps_history.keys())

    def get_fastest_worker(self) -> Optional[str]:
        """Get ID of fastest worker, or None if no data."""
        workers = self.get_active_workers()
        if not workers:
            return None
        return max(workers, key=lambda w: self.get_average_fps(w))


class HistoricalDataAnalyzer:
    """Analyzes historical BER/FER data to classify task difficulty."""

    def __init__(self, error_threshold=1e-6):
        """
        Initialize analyzer.

        Args:
            error_threshold: BER values below this are considered "zero error"
        """
        self.error_threshold = error_threshold
        self.precision_window = 0.6

    def build_system_profile(self, system: str, results: Dict) -> Dict:
        """Build frontier-aware summary data for a system."""
        system_results = results.get(system, {})
        zero_points = sorted(
            float(ebno) for ebno, ber in system_results.items() if ber == 0.0
        )
        error_points = sorted(
            float(ebno) for ebno, ber in system_results.items() if ber > 0
        )
        first_zero = zero_points[0] if zero_points else None
        last_error = error_points[-1] if error_points else None

        return {
            "has_data": bool(system_results),
            "has_zero": bool(zero_points),
            "first_zero": first_zero,
            "last_error": last_error,
            "zero_points": zero_points,
            "error_points": error_points,
        }

    def classify_task(
        self, system: str, ebno: float, results: Dict
    ) -> Tuple[str, Dict]:
        """Classify a pending task using the known frontier for its system."""
        profile = self.build_system_profile(system, results)
        if not profile["has_data"]:
            return "NEW", profile

        frontier = profile["first_zero"]
        if frontier is not None:
            distance = abs(float(ebno) - frontier)
            if distance <= self.precision_window:
                return "PRECISION", profile
            if float(ebno) < frontier:
                return "FRONTIER", profile
            return "EASY", profile

        return "EXPLORATION", profile

    def classify_ebno(
        self, system: str, ebno: float, results: Dict, fer_results: Dict
    ) -> str:
        """
        Classify difficulty of an EbNo point.

        Returns:
            'NEW': No historical data
            'HARD': Has errors (BER > threshold)
            'EASY': No errors
            'PRECISE': Very low error
        """
        if system not in results or ebno not in results[system]:
            return "NEW"

        ber = results[system][ebno]
        profile = self.build_system_profile(system, results)
        frontier = profile["first_zero"]
        if frontier is not None:
            distance = abs(float(ebno) - frontier)
            if distance <= self.precision_window:
                return "PRECISE"
            if float(ebno) < frontier and ber > 0:
                return "HARD"
            if ber == 0.0:
                return "EASY"
            return "EASY"

        # Zero error means already explored this point thoroughly
        if ber == 0.0:
            return "EASY"

        # Below threshold is considered easy
        if ber < self.error_threshold:
            return "PRECISE"

        # Has detectable errors
        return "HARD"

    def find_hard_ebno_values(self, system: str, results: Dict) -> List[float]:
        """Find all EbNo values where system has errors."""
        if system not in results:
            return []

        hard_values = []
        for ebno, ber in results[system].items():
            if ber > 0 and ber > self.error_threshold:
                hard_values.append(ebno)

        return sorted(hard_values)

    def find_easy_ebno_values(self, system: str, results: Dict) -> List[float]:
        """Find all EbNo values with zero errors."""
        if system not in results:
            return []

        easy_values = []
        for ebno, ber in results[system].items():
            if ber == 0.0:
                easy_values.append(ebno)

        return sorted(easy_values)


class SmartTaskScheduler:
    """Matches task difficulty to worker capability."""

    def __init__(self):
        self.perf_tracker = WorkerPerformanceTracker()
        self.data_analyzer = HistoricalDataAnalyzer()

    def update_worker_fps(self, worker_id: str, fps: float):
        """Update worker performance metric."""
        self.perf_tracker.update_fps(worker_id, fps)

    def select_task_for_worker(
        self,
        worker_id: str,
        pending_tasks: List[Dict],
        results: Dict,
        fer_results: Dict,
        active_workers: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Select best task from pending list for this worker.

        Strategy:
        1. Prefer idle workers getting systems not being processed elsewhere
        2. If worker is slow and has hard tasks, prefer hard tasks
        3. If worker is fast and has easy/new tasks, prefer those
        4. For new systems (all tasks NEW), only assign to fastest worker
        5. Avoid precision work on systems without zero error points

        Args:
            worker_id: ID of requesting worker
            pending_tasks: List of available tasks [{"system": str, "ebno": float}, ...]
            results: Historical BER results {system: {ebno: ber, ...}, ...}
            fer_results: Historical FER results
            active_workers: Dict of active workers with their current tasks

        Returns:
            Selected task dict or None if none suitable
        """
        if active_workers is None:
            active_workers = {}
        if not pending_tasks:
            return None

        worker_state = active_workers.get(worker_id, {})
        worker_is_idle = worker_state.get("task") is None
        worker_rank = self.perf_tracker.get_worker_rank(worker_id)
        fastest_worker = self.perf_tracker.get_fastest_worker()

        systems_in_progress = {
            info["task"].get("system")
            for wid, info in active_workers.items()
            if wid != worker_id and info.get("task")
        }

        system_profiles = {
            system: self.data_analyzer.build_system_profile(system, results)
            for system in {task["system"] for task in pending_tasks}
        }

        candidates = []
        for index, task in enumerate(pending_tasks):
            system = task["system"]
            ebno = round(float(task["ebno"]), 1)
            profile = system_profiles.get(
                system
            ) or self.data_analyzer.build_system_profile(system, results)
            category, _ = self.data_analyzer.classify_task(system, ebno, results)

            # Preserve bootstrap behavior for brand new systems.
            if (
                category == "NEW"
                and fastest_worker is not None
                and worker_id != fastest_worker
            ):
                continue

            system_busy = system in systems_in_progress
            busy_penalty = 1 if system_busy else 0

            if profile["has_zero"]:
                frontier = profile["first_zero"]
                distance = abs(ebno - frontier)

                if distance <= self.data_analyzer.precision_window:
                    # Priority 0: precision work only near the first known zero point.
                    idle_penalty = 0 if worker_is_idle else 1
                    worker_bias = (
                        0 if worker_rank == 0 else (1 if worker_rank > 0 else 2)
                    )
                    score = (
                        0,
                        busy_penalty,
                        idle_penalty,
                        distance,
                        worker_bias,
                        ebno,
                        index,
                    )
                elif ebno < frontier:
                    # Priority 1: frontier exploration on the hard side of the transition.
                    worker_bias = (
                        0 if worker_rank > 0 else (1 if worker_rank == 0 else 2)
                    )
                    score = (
                        1,
                        busy_penalty,
                        worker_bias,
                        frontier - ebno,
                        ebno if worker_rank > 0 else -ebno,
                        index,
                    )
                else:
                    # Priority 3: easy-side work far away from the frontier.
                    score = (3, busy_penalty, distance, ebno, index)
            else:
                if category == "NEW":
                    # Priority 4: bootstrap new systems.
                    score = (
                        4,
                        busy_penalty,
                        -ebno if worker_rank == 0 else ebno,
                        index,
                    )
                else:
                    # Priority 2: exploration on systems that have data but no zero point yet.
                    worker_bias = 0 if worker_rank > 0 else 1
                    ebno_bias = ebno if worker_rank > 0 else -ebno
                    score = (2, busy_penalty, worker_bias, ebno_bias, index)

            candidates.append((score, task))

        if not candidates:
            return None

        # When we have no worker history yet, keep bootstrap spreading stable.
        if not self.perf_tracker.get_active_workers():
            best_priority = min(score[0] for score, _ in candidates)
            best_bucket = [item for item in candidates if item[0][0] == best_priority]
            if len(best_bucket) == 1:
                return best_bucket[0][1]
            return best_bucket[hash(worker_id) % len(best_bucket)][1]

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _find_precision_task(
        self, worker_id: str, task_classes: List[Tuple], results: Dict
    ) -> Optional[Tuple]:
        """
        For bootstrap scenarios, find a precision/refinement task for non-fastest worker.
        Prefers low EbNo values on systems with partial data.
        """
        # Look for HARD or PRECISE tasks (refinement work)
        refinement_tasks = [
            (idx, diff, task)
            for idx, diff, task in task_classes
            if diff in ("HARD", "PRECISE", "EASY")
        ]

        if refinement_tasks:
            # Sort by EbNo ascending - lower EbNo more likely to have errors
            refinement_tasks.sort(key=lambda x: x[2]["ebno"])
            return refinement_tasks[0]

        return None
