"""Controller Registry for Comparative Microgrid Evaluation.

Provides an authoritative, typed registry of all candidate controllers across
rule-based, reinforcement learning, model predictive control, and diagnostic
oracle paradigms.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.battery_preserving import BatteryPreservingController
from src.baselines.price_aware import (
    PriceAwareConfig,
    PriceAwareController,
    load_price_aware_config,
)
from src.baselines.self_consumption import SelfConsumptionController
from src.baselines.zero_action import ZeroActionController
from src.simulation.config import load_simulation_config


ROOT = Path(__file__).resolve().parents[2]


class ControllerCategory(str, Enum):
    """Broad paradigm category of the controller."""

    RULE_BASED = "rule_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MODEL_PREDICTIVE_CONTROL = "model_predictive_control"
    DIAGNOSTIC_ORACLE = "diagnostic_oracle"


class BenchmarkStatus(str, Enum):
    """Role within the comparative benchmark suite."""

    PRIMARY_BENCHMARK = "primary_benchmark"
    DIAGNOSTIC_VARIANT = "diagnostic_variant"
    SECONDARY_BASELINE = "secondary_baseline"


class CausalityType(str, Enum):
    """Information boundary for decision making."""

    STRICTLY_CAUSAL = "strictly_causal"
    NONCAUSAL_LOOKAHEAD = "noncausal_lookahead"


@dataclass(frozen=True)
class ControllerRegistryEntry:
    """Metadata, provenance, and factory specification for a registered controller."""

    controller_id: str
    display_name: str
    category: ControllerCategory
    benchmark_status: BenchmarkStatus
    causality: CausalityType
    source_implementation: str
    original_experiment_id: str
    configuration_path: str | None
    checkpoint_reference: str | None
    observation_adapter: str
    action_adapter: str
    validation_benchmark_artifact: str
    authoritative_raw_cost_usd: float | None
    authoritative_inventory_credit_usd: float | None
    authoritative_adjusted_cost_usd: float | None
    factory: Callable[..., Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata excluding callable factory."""
        return {
            "controller_id": self.controller_id,
            "display_name": self.display_name,
            "category": self.category.value,
            "benchmark_status": self.benchmark_status.value,
            "causality": self.causality.value,
            "source_implementation": self.source_implementation,
            "original_experiment_id": self.original_experiment_id,
            "configuration_path": self.configuration_path,
            "checkpoint_reference": self.checkpoint_reference,
            "observation_adapter": self.observation_adapter,
            "action_adapter": self.action_adapter,
            "validation_benchmark_artifact": self.validation_benchmark_artifact,
            "authoritative_raw_cost_usd": self.authoritative_raw_cost_usd,
            "authoritative_inventory_credit_usd": self.authoritative_inventory_credit_usd,
            "authoritative_adjusted_cost_usd": self.authoritative_adjusted_cost_usd,
        }


# --- Factory Functions ---

def _create_zero_action() -> ZeroActionController:
    return ZeroActionController()


def _create_price_aware_continuous() -> PriceAwareController:
    return PriceAwareController()


class DiscretePriceAwareController(ZeroActionController):
    """Discrete action adapter wrapping PriceAwareController with {-150, 0, +150} kW."""

    def __init__(self, inner: PriceAwareController | None = None):
        super().__init__()
        self.inner = inner if inner is not None else PriceAwareController()

    def _reset(self) -> None:
        self.inner.reset()

    def _select_action(self, observation: Any) -> Any:
        cont_action = self.inner.select_action(observation)
        p = cont_action.requested_power_kw
        from src.simulation.actions import BatteryAction
        if p < -1e-6:
            return BatteryAction(requested_power_kw=-150.0)
        elif p > 1e-6:
            return BatteryAction(requested_power_kw=150.0)
        else:
            return BatteryAction(requested_power_kw=0.0)


def _create_price_aware_discrete() -> DiscretePriceAwareController:
    return DiscretePriceAwareController()


def _create_self_consumption() -> SelfConsumptionController:
    return SelfConsumptionController()


def _create_battery_preserving() -> BatteryPreservingController:
    return BatteryPreservingController()


def _create_ppo_continuous_control(
    checkpoint_path: str | Path | None = None,
) -> Any:
    from stable_baselines3 import PPO
    from src.evaluation.ppo_validation import FrozenPPOController

    if checkpoint_path is None:
        checkpoint_path = Path(
            r"E:\AdaptiveEnergyBrainStorage\experiments\part6_4_discounting_diagnostic_20261115\control\checkpoints\step_000016384\model.zip"
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"PPO checkpoint not found at: {checkpoint_path}")

    model = PPO.load(str(checkpoint_path))
    sim_config = load_simulation_config()
    return FrozenPPOController(model=model, config=sim_config)


def _create_ppo_discrete_diagnostic(
    checkpoint_path: str | Path | None = None,
) -> Any:
    from stable_baselines3 import PPO
    from src.evaluation.ppo_validation import FrozenPPOController

    if checkpoint_path is None:
        checkpoint_path = Path(
            r"E:\AdaptiveEnergyBrainStorage\experiments\part6_4_discrete_diagnostic_20261115_treatment\treatment\checkpoints\step_000016384\model.zip"
        )
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Discrete PPO checkpoint not found at: {checkpoint_path}")

    model = PPO.load(str(checkpoint_path))
    sim_config = load_simulation_config()
    return FrozenPPOController(model=model, config=sim_config)


def _create_seasonal_persistence_mpc(
    initial_history: Sequence[Any] | None = None,
    episode_end_utc: str = "2026-11-07T06:00:00Z",
) -> Any:
    from src.control.seasonal_persistence_mpc import SeasonalPersistenceMPCController
    from src.simulation.data_loader import load_simulation_split

    if initial_history is None:
        split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
        train_data = load_simulation_split(split_root, "train")
        initial_history = train_data[-24:]

    return SeasonalPersistenceMPCController(
        initial_history=initial_history,
        episode_end_utc=episode_end_utc,
    )


def _create_gnn_transformer_mpc(
    checkpoint_path: str | Path | None = None,
    preprocessing_path: str | Path | None = None,
    initial_history: Sequence[Any] | None = None,
    history_source: Sequence[Any] | None = None,
    episode_end_utc: str = "2026-11-07T06:00:00Z",
) -> Any:
    from src.control.gnn_transformer_mpc import (
        DEFAULT_CHECKPOINT_PATH,
        DEFAULT_PREPROCESSING_PATH,
        GNNTransformerMPCController,
    )
    from src.simulation.data_loader import load_simulation_split

    if initial_history is None:
        split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
        train_data = load_simulation_split(split_root, "train")
        initial_history = train_data[-24:]
        if history_source is None:
            history_source = train_data

    return GNNTransformerMPCController(
        checkpoint_path=checkpoint_path or DEFAULT_CHECKPOINT_PATH,
        preprocessing_path=preprocessing_path or DEFAULT_PREPROCESSING_PATH,
        initial_history=initial_history,
        history_source=history_source,
        episode_end_utc=episode_end_utc,
    )


def _create_oracle_mpc(
    partition_data: Sequence[Any] | None = None,
) -> Any:
    from src.control.oracle_mpc import OracleMPCController
    from src.simulation.data_loader import load_simulation_split

    if partition_data is None:
        split_root = ROOT / "data" / "processed" / "splits_hypothetical_campus_v1"
        partition_data = load_simulation_split(split_root, "validation")

    return OracleMPCController(partition_data=partition_data)


# --- Registry Dictionary ---

CONTROLLER_REGISTRY: dict[str, ControllerRegistryEntry] = {
    # 1. Zero Action Reference
    "zero_action": ControllerRegistryEntry(
        controller_id="zero_action",
        display_name="Zero-Action Baseline",
        category=ControllerCategory.RULE_BASED,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.baselines.zero_action.ZeroActionController",
        original_experiment_id="part4_baselines",
        configuration_path="configs/baselines.yaml",
        checkpoint_reference=None,
        observation_adapter="AgentObservation.state pass-through",
        action_adapter="Direct BatteryAction(0.0 kW)",
        validation_benchmark_artifact="results/part8_comparative_benchmark/zero_action/summary.json",
        authoritative_raw_cost_usd=43866.17,
        authoritative_inventory_credit_usd=0.00,
        authoritative_adjusted_cost_usd=43866.17,
        factory=_create_zero_action,
    ),

    # 2. Continuous Price-Aware Baseline
    "price_aware_continuous": ControllerRegistryEntry(
        controller_id="price_aware_continuous",
        display_name="Continuous Price-Aware Baseline",
        category=ControllerCategory.RULE_BASED,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.baselines.price_aware.PriceAwareController",
        original_experiment_id="part4_baselines",
        configuration_path="configs/baselines.yaml",
        checkpoint_reference=None,
        observation_adapter="AgentObservation (current-step tariffs & SOC)",
        action_adapter="Proportional linear rate scaling",
        validation_benchmark_artifact="results/part8_comparative_benchmark/price_aware_continuous/summary.json",
        authoritative_raw_cost_usd=43674.78,
        authoritative_inventory_credit_usd=29.47,
        authoritative_adjusted_cost_usd=43645.30,
        factory=_create_price_aware_continuous,
    ),

    # 3. Discrete Price-Aware Baseline
    "price_aware_discrete": ControllerRegistryEntry(
        controller_id="price_aware_discrete",
        display_name="Discrete Price-Aware Baseline",
        category=ControllerCategory.RULE_BASED,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.baselines.price_aware.PriceAwareController",
        original_experiment_id="part4_baselines_discrete",
        configuration_path="configs/baselines.yaml",
        checkpoint_reference=None,
        observation_adapter="AgentObservation (current-step tariffs & SOC)",
        action_adapter="Fixed step action (±150 kW)",
        validation_benchmark_artifact="results/part8_comparative_benchmark/price_aware_discrete/summary.json",
        authoritative_raw_cost_usd=43662.24,
        authoritative_inventory_credit_usd=29.47,
        authoritative_adjusted_cost_usd=43632.76,
        factory=_create_price_aware_discrete,
    ),

    # 4. Continuous PPO Diagnostic Checkpoint (step 16k)
    "ppo_continuous_control": ControllerRegistryEntry(
        controller_id="ppo_continuous_control",
        display_name="PPO Continuous Control (16k steps)",
        category=ControllerCategory.REINFORCEMENT_LEARNING,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.evaluation.ppo_validation.FrozenPPOController",
        original_experiment_id="part6_4_discounting_diagnostic_20261115",
        configuration_path="configs/ppo_discounting_diagnostic.yaml",
        checkpoint_reference="experiments/part6_4_discounting_diagnostic_20261115/control/checkpoints/step_000016384/model.zip",
        observation_adapter="ObservationEncoder (15-dim normalized vector)",
        action_adapter="map_normalized_action (Box(1) continuous mapping)",
        validation_benchmark_artifact="experiments/part6_4_discounting_diagnostic_20261115/validation_16384/validation.json",
        authoritative_raw_cost_usd=43835.77,
        authoritative_inventory_credit_usd=-29.47,
        authoritative_adjusted_cost_usd=43865.25,
        factory=_create_ppo_continuous_control,
    ),

    # 5. Seasonal Persistence MPC
    "mpc_seasonal_persistence": ControllerRegistryEntry(
        controller_id="mpc_seasonal_persistence",
        display_name="Seasonal Persistence MPC",
        category=ControllerCategory.MODEL_PREDICTIVE_CONTROL,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.control.seasonal_persistence_mpc.SeasonalPersistenceMPCController",
        original_experiment_id="part7_3_seasonal_persistence_mpc",
        configuration_path="configs/mpc_controller.yaml",
        checkpoint_reference=None,
        observation_adapter="CausalHistoricalRecord (24h seasonal lag ingestion)",
        action_adapter="Rolling 24h HiGHS LP first action",
        validation_benchmark_artifact="results/part7_3_seasonal_persistence_mpc/seasonal_persistence_summary.json",
        authoritative_raw_cost_usd=43759.37,
        authoritative_inventory_credit_usd=12.53,
        authoritative_adjusted_cost_usd=43746.85,
        factory=_create_seasonal_persistence_mpc,
    ),

    # 6. GNN-Transformer MPC
    "mpc_gnn_transformer": ControllerRegistryEntry(
        controller_id="mpc_gnn_transformer",
        display_name="GNN-Transformer MPC",
        category=ControllerCategory.MODEL_PREDICTIVE_CONTROL,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.control.gnn_transformer_mpc.GNNTransformerMPCController",
        original_experiment_id="part7_4_gnn_mpc",
        configuration_path="configs/mpc_controller.yaml",
        checkpoint_reference="checkpoints/part5_9-colab-20261111T172326Z/best.pt",
        observation_adapter="NodeHistoricalRecord (8-node historical scaling)",
        action_adapter="Rolling 24h HiGHS LP first action",
        validation_benchmark_artifact="results/part7_4_gnn_mpc/gnn_mpc_validation_summary.json",
        authoritative_raw_cost_usd=43715.40,
        authoritative_inventory_credit_usd=12.53,
        authoritative_adjusted_cost_usd=43702.87,
        factory=_create_gnn_transformer_mpc,
    ),

    # 7. 24h Rolling Perfect-Foresight Oracle MPC
    "mpc_oracle_rolling_24h": ControllerRegistryEntry(
        controller_id="mpc_oracle_rolling_24h",
        display_name="Oracle MPC (24h Rolling Perfect-Foresight)",
        category=ControllerCategory.DIAGNOSTIC_ORACLE,
        benchmark_status=BenchmarkStatus.PRIMARY_BENCHMARK,
        causality=CausalityType.NONCAUSAL_LOOKAHEAD,
        source_implementation="src.control.oracle_mpc.OracleMPCController",
        original_experiment_id="part7_2_oracle_mpc",
        configuration_path="configs/mpc_controller.yaml",
        checkpoint_reference=None,
        observation_adapter="Noncausal partition lookup (24h future realization)",
        action_adapter="Rolling 24h HiGHS LP first action",
        validation_benchmark_artifact="results/part7_2_oracle_mpc/oracle_validation_summary.json",
        authoritative_raw_cost_usd=43409.05,
        authoritative_inventory_credit_usd=12.53,
        authoritative_adjusted_cost_usd=43396.52,
        factory=_create_oracle_mpc,
    ),

    # 8. Discrete PPO Diagnostic Variant (step 16k)
    "ppo_discrete_diagnostic": ControllerRegistryEntry(
        controller_id="ppo_discrete_diagnostic",
        display_name="PPO Discrete Diagnostic (16k steps)",
        category=ControllerCategory.REINFORCEMENT_LEARNING,
        benchmark_status=BenchmarkStatus.DIAGNOSTIC_VARIANT,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.evaluation.ppo_validation.FrozenPPOController",
        original_experiment_id="part6_4_discrete_diagnostic_20261115",
        configuration_path="configs/ppo_discrete_diagnostic.yaml",
        checkpoint_reference="experiments/part6_4_discrete_diagnostic_20261115_treatment/treatment/checkpoints/step_000016384/model.zip",
        observation_adapter="ObservationEncoder (15-dim normalized vector)",
        action_adapter="DISCRETE_ACTION_MAP (Discrete(3) mapping)",
        validation_benchmark_artifact="experiments/part6_4_discrete_diagnostic_20261115_treatment/validation_16384/validation.json",
        authoritative_raw_cost_usd=43835.77,
        authoritative_inventory_credit_usd=-29.47,
        authoritative_adjusted_cost_usd=43865.25,
        factory=_create_ppo_discrete_diagnostic,
    ),

    # 9. Self-Consumption Baseline
    "self_consumption": ControllerRegistryEntry(
        controller_id="self_consumption",
        display_name="Self-Consumption Baseline",
        category=ControllerCategory.RULE_BASED,
        benchmark_status=BenchmarkStatus.SECONDARY_BASELINE,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.baselines.self_consumption.SelfConsumptionController",
        original_experiment_id="part4_baselines",
        configuration_path="configs/baselines.yaml",
        checkpoint_reference=None,
        observation_adapter="AgentObservation (instantaneous renewable surplus/deficit)",
        action_adapter="Net load balance tracking",
        validation_benchmark_artifact="results/baseline_comparison/summary.json",
        authoritative_raw_cost_usd=None,
        authoritative_inventory_credit_usd=None,
        authoritative_adjusted_cost_usd=None,
        factory=_create_self_consumption,
    ),

    # 10. Battery-Preserving Baseline
    "battery_preserving": ControllerRegistryEntry(
        controller_id="battery_preserving",
        display_name="Battery-Preserving Baseline",
        category=ControllerCategory.RULE_BASED,
        benchmark_status=BenchmarkStatus.SECONDARY_BASELINE,
        causality=CausalityType.STRICTLY_CAUSAL,
        source_implementation="src.baselines.battery_preserving.BatteryPreservingController",
        original_experiment_id="part4_baselines",
        configuration_path="configs/baselines.yaml",
        checkpoint_reference=None,
        observation_adapter="AgentObservation (instantaneous price & SOC deadband)",
        action_adapter="Restricted throughput dispatch",
        validation_benchmark_artifact="results/baseline_comparison/summary.json",
        authoritative_raw_cost_usd=None,
        authoritative_inventory_credit_usd=None,
        authoritative_adjusted_cost_usd=None,
        factory=_create_battery_preserving,
    ),
}


def get_controller_entry(controller_id: str) -> ControllerRegistryEntry:
    """Retrieve a registered controller entry by unique ID."""
    if controller_id not in CONTROLLER_REGISTRY:
        available = list(CONTROLLER_REGISTRY.keys())
        raise KeyError(
            f"Controller '{controller_id}' not found in registry. Available: {available}"
        )
    return CONTROLLER_REGISTRY[controller_id]


def list_controllers(
    category: ControllerCategory | None = None,
    status: BenchmarkStatus | None = None,
    causality: CausalityType | None = None,
) -> list[ControllerRegistryEntry]:
    """Filter registered controllers by category, benchmark status, or causality."""
    results = []
    for entry in CONTROLLER_REGISTRY.values():
        if category is not None and entry.category != category:
            continue
        if status is not None and entry.benchmark_status != status:
            continue
        if causality is not None and entry.causality != causality:
            continue
        results.append(entry)
    return results


def get_primary_benchmark_ids() -> list[str]:
    """Return the canonical list of primary benchmark controller IDs."""
    return [
        entry.controller_id
        for entry in list_controllers(status=BenchmarkStatus.PRIMARY_BENCHMARK)
    ]
