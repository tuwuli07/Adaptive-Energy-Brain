"""Phase 8.3 Computational Feasibility and Latency Profiling Engine.

Profiles runtime performance, per-step decision latencies, memory footprint,
and solver convergence across rule-based, DRL, and MPC controllers on the
1,313-hour validation split under isolated, reproducible conditions.
"""

import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch

from scripts.run_simulation import _components
from src.control.mpc_solver import MPCSolverDiagnostics
from src.evaluation.controller_registry import (
    CONTROLLER_REGISTRY,
    CausalityType,
    ControllerCategory,
    get_controller_entry,
    get_primary_benchmark_ids,
)
from src.simulation.config import load_simulation_config
from src.simulation.data_loader import load_simulation_split
from src.simulation.environment import MicrogridEnvironment
from src.simulation.metrics import load_cost_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_ROOT = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
DEFAULT_ASSETS_PATH = ROOT / "configs" / "synthetic_scenario.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "part8_comparative_benchmark"


# Windows Memory Structs via ctypes
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_process_memory_mb() -> tuple[float, float]:
    """Return (current_rss_mb, peak_rss_mb) for the current process."""
    if platform.system() == "Windows":
        try:
            pmc = PROCESS_MEMORY_COUNTERS_EX()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            hProcess = kernel32.GetCurrentProcess()
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            res = psapi.GetProcessMemoryInfo(hProcess, ctypes.byref(pmc), pmc.cb)
            if res:
                return (pmc.WorkingSetSize / (1024**2), pmc.PeakWorkingSetSize / (1024**2))
        except Exception:
            pass
    return (0.0, 0.0)


def get_hardware_and_environment_specs() -> dict[str, Any]:
    """Capture comprehensive system, processor, OS, and runtime configurations."""
    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    if platform.system() == "Windows":
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram_gb = stat.ullTotalPhys / (1024**3)
            avail_ram_gb = stat.ullAvailPhys / (1024**3)
        except Exception:
            pass

    clock_info = time.get_clock_info("perf_counter")

    return {
        "operating_system": f"{platform.system()} {platform.release()} (Build {platform.version()})",
        "platform_string": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(avail_ram_gb, 2),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "pytorch_cpu_threads": torch.get_num_threads(),
        "pytorch_cuda_available": torch.cuda.is_available(),
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "timer_clock": {
            "implementation": clock_info.implementation,
            "resolution_seconds": clock_info.resolution,
            "monolithic": clock_info.monotonic,
        },
    }


@dataclass(frozen=True)
class LatencyPercentiles:
    """Latency distribution percentiles in milliseconds."""

    mean_ms: float
    std_ms: float
    min_ms: float
    p25_ms: float
    p50_median_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_latencies_ms(cls, latencies: Sequence[float]) -> "LatencyPercentiles":
        if not latencies:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(latencies, dtype=np.float64)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            p25_ms=float(np.percentile(arr, 25)),
            p50_median_ms=float(np.percentile(arr, 50)),
            p75_ms=float(np.percentile(arr, 75)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(np.max(arr)),
        )


@dataclass(frozen=True)
class ControllerProfilingResult:
    """Complete profiling and resource consumption profile for a controller."""

    controller_id: str
    display_name: str
    category: str
    causality: str
    total_decisions: int
    warmup_steps_executed: int
    initialization_time_ms: float
    total_decision_latency: LatencyPercentiles
    simulator_stepping_latency: LatencyPercentiles
    total_evaluation_elapsed_s: float
    baseline_memory_rss_mb: float
    peak_memory_rss_mb: float
    incremental_memory_mb: float
    errors_or_fallbacks: int
    # Component breakdown for GNN MPC
    neural_inference_latency: LatencyPercentiles | None = None
    lp_solve_latency: LatencyPercentiles | None = None
    # Solver iteration stats for MPC
    solver_iterations_mean: float | None = None
    solver_iterations_max: int | None = None
    solver_status_codes: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_controller(
    controller_id: str,
    *,
    partition: str = "validation",
    split_root: Path | str = DEFAULT_SPLIT_ROOT,
    assets_path: Path | str = DEFAULT_ASSETS_PATH,
    rating_origin: str = "hypothetical_example",
    warmup_steps: int = 5,
) -> ControllerProfilingResult:
    """Execute rigorous, isolated CPU-based latency and resource profiling."""
    if partition == "test":
        raise PermissionError("Access to held-out test partition is strictly blocked.")

    entry = get_controller_entry(controller_id)
    split_root_p = Path(split_root)
    assets_path_p = Path(assets_path)

    # 1. Load validation partition data
    data = load_simulation_split(split_root_p, partition)
    sim_config = load_simulation_config()
    cost_config = load_cost_config()
    renewables, demand = _components(data, assets_path_p, rating_origin)

    # Force garbage collection and capture baseline memory
    gc.collect()
    mem_base_rss, _ = get_process_memory_mb()

    # 2. Measure initialization time
    t0_init = time.perf_counter_ns()
    if controller_id == "mpc_oracle_rolling_24h":
        controller = entry.factory(partition_data=data)
    elif controller_id == "mpc_gnn_transformer" and partition == "validation":
        train_data = load_simulation_split(split_root_p, "train")
        controller = entry.factory(initial_history=train_data[-24:], history_source=train_data)
    elif controller_id == "mpc_seasonal_persistence" and partition == "validation":
        train_data = load_simulation_split(split_root_p, "train")
        controller = entry.factory(initial_history=train_data[-24:])
    else:
        controller = entry.factory()
    init_time_ms = (time.perf_counter_ns() - t0_init) / 1e6

    mem_after_init_rss, _ = get_process_memory_mb()

    # 3. Warm-up phase on a dedicated throwaway environment to warm up JIT, BLAS, and LP matrix caches
    if warmup_steps > 0:
        from src.simulation.data_loader import SimulationData
        dummy_slice = SimulationData(data[:5], data.metadata)
        dummy_renewables, dummy_demand = _components(dummy_slice, assets_path_p, rating_origin)
        dummy_env = MicrogridEnvironment(
            dummy_slice,
            dummy_renewables,
            dummy_demand,
            sim_config,
            cost_config,
        )
        w_obs = dummy_env.reset()
        for _ in range(warmup_steps):
            controller.select_action(w_obs)

        # Clean reset of controller state after warm-up
        controller.reset()

    # 4. Create fresh, dedicated benchmark environment (untouched by warm-up)
    env = MicrogridEnvironment(
        data,
        renewables,
        demand,
        sim_config,
        cost_config,
    )
    env.reset()

    # Methodological assertions: verify complete state isolation before starting measurement
    assert env._index == 1, f"Replay starting index must be 1, found {env._index}"
    assert not env.terminated, "Environment must not be terminated before replay"
    if hasattr(controller, "step_actions"):
        assert len(controller.step_actions) == 0, "Controller actions must be empty before replay"
    if hasattr(controller, "step_diagnostics"):
        assert len(controller.step_diagnostics) == 0, "Controller diagnostics must be empty before replay"
    if hasattr(controller, "fallback_count"):
        assert controller.fallback_count == 0, "Controller fallback count must be zero before replay"

    decision_latencies_ms: list[float] = []
    stepping_latencies_ms: list[float] = []
    inference_latencies_ms: list[float] = []
    solve_latencies_ms: list[float] = []
    solver_iterations: list[int] = []
    solver_statuses: dict[str, int] = {}

    t_eval_start = time.perf_counter()

    while not env.terminated:
        obs = env.observation

        # Measure pure controller decision latency (isolated from stepping)
        t_dec_start = time.perf_counter_ns()
        action = controller.select_action(obs)
        t_dec_end = time.perf_counter_ns()
        dec_ms = (t_dec_end - t_dec_start) / 1e6
        decision_latencies_ms.append(dec_ms)

        # Measure simulator stepping latency
        t_step_start = time.perf_counter_ns()
        step = env.step(action)
        t_step_end = time.perf_counter_ns()
        step_ms = (t_step_end - t_step_start) / 1e6
        stepping_latencies_ms.append(step_ms)

    total_eval_elapsed_s = time.perf_counter() - t_eval_start
    _, peak_mem_rss = get_process_memory_mb()
    incremental_mem_mb = max(0.0, peak_mem_rss - mem_base_rss)

    # Extract component-level timings for GNN-MPC and MPC variants
    inference_p: LatencyPercentiles | None = None
    solve_p: LatencyPercentiles | None = None
    iter_mean: float | None = None
    iter_max: int | None = None
    status_summary: dict[str, int] | None = None

    if hasattr(controller, "inference_times") and controller.inference_times:
        inf_ms = [t * 1000.0 for t in controller.inference_times]
        inference_p = LatencyPercentiles.from_latencies_ms(inf_ms)
    elif hasattr(controller, "step_diagnostics") and controller.step_diagnostics:
        inf_s = [getattr(d, "inference_time_seconds", None) for d in controller.step_diagnostics]
        inf_s = [t for t in inf_s if t is not None]
        if inf_s:
            inference_p = LatencyPercentiles.from_latencies_ms([t * 1000.0 for t in inf_s])

    if hasattr(controller, "solve_times") and controller.solve_times:
        sol_ms = [t * 1000.0 for t in controller.solve_times]
        solve_p = LatencyPercentiles.from_latencies_ms(sol_ms)
    elif hasattr(controller, "step_diagnostics") and controller.step_diagnostics:
        sol_s = [getattr(d, "solve_time_seconds", None) for d in controller.step_diagnostics]
        sol_s = [t for t in sol_s if t is not None]
        if sol_s:
            solve_p = LatencyPercentiles.from_latencies_ms([t * 1000.0 for t in sol_s])

    if hasattr(controller, "step_diagnostics") and controller.step_diagnostics:
        for d in controller.step_diagnostics:
            st = getattr(d, "status", None)
            if st is not None:
                st_code = f"HighsStatus_{st}"
                solver_statuses[st_code] = solver_statuses.get(st_code, 0) + 1
        if solver_statuses:
            status_summary = solver_statuses

    fallback_count = getattr(controller, "fallback_count", 0)

    return ControllerProfilingResult(
        controller_id=controller_id,
        display_name=entry.display_name,
        category=entry.category.value,
        causality=entry.causality.value,
        total_decisions=len(decision_latencies_ms),
        warmup_steps_executed=warmup_steps,
        initialization_time_ms=round(init_time_ms, 2),
        total_decision_latency=LatencyPercentiles.from_latencies_ms(decision_latencies_ms),
        simulator_stepping_latency=LatencyPercentiles.from_latencies_ms(stepping_latencies_ms),
        total_evaluation_elapsed_s=round(total_eval_elapsed_s, 2),
        baseline_memory_rss_mb=round(mem_base_rss, 2),
        peak_memory_rss_mb=round(peak_mem_rss, 2),
        incremental_memory_mb=round(incremental_mem_mb, 2),
        errors_or_fallbacks=fallback_count,
        neural_inference_latency=inference_p,
        lp_solve_latency=solve_p,
        solver_iterations_mean=round(iter_mean, 2) if iter_mean is not None else None,
        solver_iterations_max=iter_max,
        solver_status_codes=status_summary,
    )


def run_profiling_suite(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    warmup_steps: int = 5,
    isolated_subprocesses: bool = True,
) -> dict[str, Any]:
    """Execute complete reproducible CPU profiling suite across all primary controllers.
    
    When isolated_subprocesses=True, each controller is profiled in an independent child
    Python process to eliminate cross-controller memory retention and guarantee true
    standalone working-set (RSS) attribution and clean heap/cache boundaries.
    """
    sys_specs = get_hardware_and_environment_specs()
    primary_ids = get_primary_benchmark_ids()

    print("=" * 80)
    print("PHASE 8.3 COMPUTATIONAL FEASIBILITY & LATENCY PROFILING SUITE")
    print("=" * 80)
    print(f"OS: {sys_specs['operating_system']}")
    print(f"Processor: {sys_specs['processor']}")
    print(f"Logical CPUs: {sys_specs['cpu_count_logical']} | PyTorch Threads: {sys_specs['pytorch_cpu_threads']}")
    print(f"Total Physical RAM: {sys_specs['total_ram_gb']} GB")
    print(f"Warm-up steps per controller: {warmup_steps} (state-isolated)")
    print(f"Process Execution Mode: {'Isolated Subprocesses' if isolated_subprocesses else 'In-Process Sequential'}")
    print("-" * 80)

    results: dict[str, Any] = {}
    summary_table: list[dict[str, Any]] = []

    child_script_template = """
import json
import sys
from pathlib import Path

ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))

from src.evaluation.profiling import profile_controller

cid = "{cid}"
res = profile_controller(cid, warmup_steps={warmup_steps})
data = res.to_dict()
print("JSON_START" + json.dumps(data) + "JSON_END")
"""

    for cid in primary_ids:
        entry = get_controller_entry(cid)
        print(f"\nProfiling controller: {entry.display_name} ({cid})...")
        t_start = time.perf_counter()

        if isolated_subprocesses:
            import subprocess
            code = child_script_template.format(
                root=str(ROOT),
                cid=cid,
                warmup_steps=warmup_steps,
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Subprocess profiling failed for {cid}:\n{proc.stderr}")
            output = proc.stdout
            payload = output.split("JSON_START")[1].split("JSON_END")[0]
            prof_dict = json.loads(payload)
        else:
            prof = profile_controller(cid, warmup_steps=warmup_steps)
            prof_dict = prof.to_dict()

        elapsed = time.perf_counter() - t_start

        p = prof_dict["total_decision_latency"]
        inf_p = prof_dict.get("neural_inference_latency")
        sol_p = prof_dict.get("lp_solve_latency")

        print(f"  -> Init Time: {prof_dict['initialization_time_ms']:.2f} ms")
        print(f"  -> Mean Latency: {p['mean_ms']:.3f} ms | Median (P50): {p['p50_median_ms']:.3f} ms")
        print(f"  -> P95: {p['p95_ms']:.3f} ms | P99: {p['p99_ms']:.3f} ms | Max: {p['max_ms']:.3f} ms")
        print(f"  -> Total Replay Time: {prof_dict['total_evaluation_elapsed_s']:.2f} s | Standalone Peak RSS: {prof_dict['peak_memory_rss_mb']:.1f} MB (Base: {prof_dict['baseline_memory_rss_mb']:.1f} MB, Net: {prof_dict['incremental_memory_mb']:.1f} MB)")
        if inf_p:
            print(f"     [GNN Neural Inference]: Mean={inf_p['mean_ms']:.2f} ms, P50={inf_p['p50_median_ms']:.2f} ms, P95={inf_p['p95_ms']:.2f} ms")
        if sol_p:
            print(f"     [HiGHS LP Solving]:   Mean={sol_p['mean_ms']:.2f} ms, P50={sol_p['p50_median_ms']:.2f} ms, P95={sol_p['p95_ms']:.2f} ms")

        row = {
            "controller_id": cid,
            "display_name": prof_dict["display_name"],
            "category": prof_dict["category"],
            "causality": prof_dict["causality"],
            "init_time_ms": prof_dict["initialization_time_ms"],
            "mean_latency_ms": round(p["mean_ms"], 3),
            "median_p50_ms": round(p["p50_median_ms"], 3),
            "p95_latency_ms": round(p["p95_ms"], 3),
            "p99_latency_ms": round(p["p99_ms"], 3),
            "max_latency_ms": round(p["max_ms"], 3),
            "total_elapsed_s": prof_dict["total_evaluation_elapsed_s"],
            "baseline_rss_mb": prof_dict["baseline_memory_rss_mb"],
            "standalone_peak_rss_mb": prof_dict["peak_memory_rss_mb"],
            "incremental_mem_mb": prof_dict["incremental_memory_mb"],
            "errors_or_fallbacks": prof_dict["errors_or_fallbacks"],
            "inference_mean_ms": round(inf_p["mean_ms"], 2) if inf_p else None,
            "lp_solve_mean_ms": round(sol_p["mean_ms"], 2) if sol_p else None,
            "solver_iterations_mean": prof_dict.get("solver_iterations_mean"),
        }
        summary_table.append(row)
        results[cid] = prof_dict

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    suite_artifact = {
        "benchmark_partition": "validation",
        "decisions_per_controller": 1313,
        "execution_mode": "isolated_subprocesses",
        "hardware_and_environment": sys_specs,
        "memory_attribution_methodology": {
            "description": "Each controller profiled in an independent child process. Peak RSS reflects true standalone process working set. Baseline RSS captures Python runtime and shared libraries prior to controller instantiation.",
            "metrics_defined": [
                "baseline_rss_mb: Win32 WorkingSetSize of Python process with standard libraries before controller creation",
                "standalone_peak_rss_mb: Win32 PeakWorkingSetSize reached during 1,313-step benchmark execution in isolated process",
                "incremental_mem_mb: max(0.0, standalone_peak_rss_mb - baseline_rss_mb) attributed specifically to model weights, graph data, and solver buffers"
            ]
        },
        "controllers": results,
        "summary_table": summary_table,
    }

    out_file = out_root / "profiling_metrics.json"
    out_file.write_text(json.dumps(suite_artifact, indent=2), encoding="utf-8")
    print(f"\nProfiling metrics written to: {out_file}")

    return suite_artifact


if __name__ == "__main__":
    run_profiling_suite()
