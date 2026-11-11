"""Run final held-out evaluation for a completed Part 5.8 GNN-Transformer.

Example:
    python scripts/evaluate_forecaster.py --checkpoint <best.pt> \
        --training-metadata <experiment.json> --device cpu
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.evaluation import (
    EvaluationConfig,
    ForecastEvaluationError,
    run_forecast_evaluation,
    save_evaluation_artifacts,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ForecastEvaluationError(f"Cannot load YAML configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ForecastEvaluationError(f"Configuration must be a YAML mapping: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--preprocessing", type=Path, default=None)
    parser.add_argument("--training-metadata", type=Path, default=None)
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/forecasting_evaluation.yaml",
    )
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs/forecasting_model.yaml"
    )
    parser.add_argument("--sequence-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--horizon-hours", type=int)
    parser.add_argument("--seasonal-period-hours", type=int)
    parser.add_argument("--example-windows", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        model_config_path = _resolve(ROOT, args.model_config)
        model_config = _read_yaml(model_config_path)
        evaluation_yaml = _read_yaml(_resolve(ROOT, args.evaluation_config))
        evaluation_config = EvaluationConfig.from_mapping(evaluation_yaml)
        overrides = {
            "batch_size": args.batch_size,
            "device": args.device,
            "forecast_horizon_hours": args.horizon_hours,
            "seasonal_period_hours": args.seasonal_period_hours,
            "representative_window_count": args.example_windows,
        }
        evaluation_config = replace(
            evaluation_config,
            **{name: value for name, value in overrides.items() if value is not None},
        )
        evaluation_config.validate()

        model_data = model_config.get("data")
        if not isinstance(model_data, dict) or not isinstance(
            model_data.get("split_bundle"), str
        ):
            raise ForecastEvaluationError("Model configuration requires data.split_bundle.")
        split_root = _resolve(
            ROOT,
            args.split_root
            or os.environ.get("AEB_SPLIT_ROOT")
            or model_data["split_bundle"],
        )
        preprocessing_path = _resolve(
            ROOT,
            args.preprocessing or split_root / "preprocessing.json",
        )
        sequence_value = (
            args.sequence_config
            or model_data.get("sequence_config")
            or "configs/forecasting_data.yaml"
        )
        sequence_config_path = _resolve(ROOT, sequence_value)
        sequence_config = _read_yaml(sequence_config_path)
        checkpoint = _resolve(ROOT, args.checkpoint)
        training_metadata = (
            _resolve(ROOT, args.training_metadata)
            if args.training_metadata is not None
            else None
        )

        output = run_forecast_evaluation(
            checkpoint_path=checkpoint,
            split_root=split_root,
            preprocessing_path=preprocessing_path,
            model_config=model_config,
            sequence_config=sequence_config,
            evaluation_config=evaluation_config,
            training_metadata_path=training_metadata,
        )
        experiment_id = output.results["experiment_id"]
        if args.output_dir is not None:
            output_dir = _resolve(ROOT, args.output_dir)
        else:
            paths = evaluation_yaml.get("paths")
            if not isinstance(paths, dict) or not isinstance(paths.get("results_dir"), str):
                raise ForecastEvaluationError(
                    "Evaluation configuration requires paths.results_dir."
                )
            external_experiments = os.environ.get("AEB_EXPERIMENTS_ROOT")
            external_evaluations = os.environ.get("AEB_EVALUATIONS_ROOT")
            if external_evaluations:
                results_root = _resolve(ROOT, external_evaluations)
            elif external_experiments:
                results_root = _resolve(ROOT, external_experiments) / "evaluations"
            else:
                results_root = _resolve(ROOT, paths["results_dir"])
            output_dir = results_root / experiment_id
        save_evaluation_artifacts(
            output,
            output_dir,
            representative_window_count=evaluation_config.representative_window_count,
        )
        metrics = output.results["evaluation"]["metrics"]["overall"]
        print(f"Experiment: {experiment_id}")
        print(
            f"Test windows: {output.results['evaluation']['evaluated_sample_count']} | "
            f"horizon: {output.results['dataset']['forecast_horizon_hours']}h | "
            f"synthetic: {output.results['dataset']['synthetic']}"
        )
        for method, values in metrics.items():
            print(
                f"{method}: MAE={values['mae_kw']:.4f} kW, "
                f"RMSE={values['rmse_kw']:.4f} kW, "
                f"n={values['valid_observation_count']}"
            )
        print(f"Evaluation artifacts: {output_dir}")
        return 0
    except (ForecastEvaluationError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
