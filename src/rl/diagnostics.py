"""Read-only PPO telemetry using callbacks and removable PyTorch hooks.

Supports the installed SB3 2.7.1 CPU MlpPolicy path. No optimizer, reward,
sampling, or installed-library implementation is replaced.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class PPODiagnostics(BaseCallback):
    """One-environment telemetry; hooks observe actual backward/optimizer calls."""

    def __init__(self, output, reward_scale=1.0):
        super().__init__()
        self.output = Path(output)
        if (isinstance(reward_scale, bool) or not isinstance(reward_scale, (int, float))
                or not math.isfinite(reward_scale) or reward_scale <= 0):
            raise ValueError("reward_scale must be a finite positive number.")
        self.reward_scale = float(reward_scale)
        self.output.mkdir()
        self.handles = []
        self.pending_terminal = None
        self.rows = []
        self.gradient_squares = {}
        self.parameter_snapshots = None
        self.gradient_step = 0
        self.transition_count = 0

    def emit(self, name, record):
        # JSON rejects NaN/Inf rather than silently dropping bad observations.
        content = json.dumps(record, allow_nan=False)
        with (self.output / (name + ".jsonl")).open("a", encoding="utf-8") as stream:
            stream.write(content + "\n")

    def attach(self, model):
        if model.n_envs != 1 or model.policy.squash_output or model.use_sde:
            raise ValueError("Diagnostics require one unsquashed Gaussian PPO environment.")
        self.model = model
        self.parameters = dict(model.policy.named_parameters())
        self.groups = {}
        for name, parameter in self.parameters.items():
            if name.startswith(("mlp_extractor.policy_net.", "action_net.")) or name == "log_std":
                group = "actor"
            elif name.startswith(("mlp_extractor.value_net.", "value_net.")):
                group = "critic"
            else:
                raise ValueError(f"Unsupported shared/trainable parameter: {name}")
            self.groups[name] = group
            def capture(gradient, name=name):
                if name in self.gradient_squares:
                    raise RuntimeError("Unexpected gradient accumulation in PPO diagnostics.")
                self.gradient_squares[name] = float(gradient.detach().double().square().sum())
                return None  # Observe only; never replace the gradient.
            self.handles.append(parameter.register_hook(capture))
        self.handles.append(model.policy.optimizer.register_step_pre_hook(self._before_optimizer))
        self.handles.append(model.policy.optimizer.register_step_post_hook(self._after_optimizer))
        self.handles.append(model.policy.value_net.register_forward_hook(self._terminal_value))

    def _before_optimizer(self, optimizer, args, kwargs):
        if set(self.gradient_squares) != set(self.parameters):
            raise RuntimeError("Missing backward gradient observations.")
        record = {"optimizer_step": self.gradient_step + 1, "timesteps": self.model.num_timesteps,
                  "max_grad_norm": self.model.max_grad_norm}
        for group in ("actor", "critic", "global"):
            names = [n for n in self.parameters if group == "global" or self.groups[n] == group]
            pre = math.sqrt(sum(self.gradient_squares[n] for n in names))
            post = math.sqrt(sum(float(self.parameters[n].grad.detach().double().square().sum()) for n in names))
            record[group + "_pre_norm"] = pre
            record[group + "_post_norm"] = post
        record["observed_global_scale"] = (record["global_post_norm"] / record["global_pre_norm"]
                                            if record["global_pre_norm"] else 1.0)
        record["clipped"] = record["global_pre_norm"] > self.model.max_grad_norm
        if record["global_post_norm"] > self.model.max_grad_norm + 1e-5:
            raise ValueError("Observed gradient exceeds configured global clipping limit.")
        self.parameter_snapshots = {
            name: parameter.detach().clone() for name, parameter in self.parameters.items()
        }
        self.emit("gradients", record)
        self.gradient_step += 1
        self.gradient_squares.clear()

    def _after_optimizer(self, optimizer, args, kwargs):
        if self.parameter_snapshots is None:
            raise RuntimeError("Missing pre-step parameter snapshots.")
        record = {"optimizer_step": self.gradient_step, "timesteps": self.model.num_timesteps}
        for group in ("actor", "critic", "global"):
            names = [name for name in self.parameters
                     if group == "global" or self.groups[name] == group]
            squared_delta = sum(
                float((self.parameters[name].detach() - self.parameter_snapshots[name])
                      .double().square().sum())
                for name in names
            )
            record[group + "_update_norm"] = math.sqrt(squared_delta)
        self.emit("parameter_updates", record)
        self.parameter_snapshots = None

    def _terminal_value(self, module, inputs, output):
        if self.pending_terminal is not None:
            # SB3's actual predict_values call after the callback, before its
            # timeout reward adjustment. No duplicate or approximate prediction.
            self.pending_terminal["terminal_bootstrap_value"] = float(output.detach().flatten()[0])
            self.pending_terminal["dollar_equivalent_terminal_bootstrap_value"] = (
                self.pending_terminal["terminal_bootstrap_value"] / self.reward_scale
            )
            self.pending_terminal = None

    def _on_rollout_start(self):
        if self.pending_terminal is not None:
            raise RuntimeError("Unobserved terminal bootstrap.")
        self.rows = []

    def _on_step(self):
        local = self.locals
        info = local["infos"][0]
        before = self.model._last_obs[0].copy()
        after = (info["terminal_observation"] if local["dones"][0] else local["new_obs"][0])
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(
                torch.as_tensor(before[None], device=self.model.device))
            if hasattr(distribution, "distribution") and hasattr(distribution.distribution, "probs"):
                probs = distribution.distribution.probs.detach().cpu().numpy().flatten().tolist()
                mean = float(probs[2] - probs[0])
                std = float(np.sqrt(probs[0] * (-1.0 - mean)**2 + probs[1] * (0.0 - mean)**2 + probs[2] * (1.0 - mean)**2))
                entropy = float(distribution.entropy().flatten()[0])
            else:
                mean = float(distribution.distribution.mean.flatten()[0])
                std = float(distribution.distribution.stddev.flatten()[0])
                entropy = float(distribution.entropy().flatten()[0])
        sampled = float(local["actions"].flatten()[0])
        clipped = float(local["clipped_actions"].flatten()[0])
        cost = float(info["step_metrics"]["values"]["net_operating_cost"])
        raw_reward = float(np.float32(-cost))
        shaping_usd = float(info.get("shaping_reward_usd", 0.0))
        effective_unscaled_reward = raw_reward + shaping_usd
        scaled_reward = float(local["rewards"][0])
        expected_scaled_reward = float(np.float32(effective_unscaled_reward * self.reward_scale))
        if not np.isclose(scaled_reward, expected_scaled_reward, rtol=1e-5, atol=1e-6):
            raise ValueError("PPO vector reward differs from the configured scale of committed accounting.")
        value_prediction = float(local["values"].flatten()[0])
        row = {"timesteps": self.model.num_timesteps, "observation": before.tolist(),
               "observed_import_price": float(before[6]), "observed_export_price": float(before[7]),
               "soc_before": float(before[0]), "soc_after": float(after[0]),
               "decision_timestamp_utc": info["observed_interval_utc"],
               "policy_mean": mean, "policy_std": std, "policy_entropy": entropy,
               "sampled_action": sampled, "bounded_action": clipped,
               "action_clipped": bool(sampled != clipped),
               "requested_kw": info["requested_power_kw"], "executed_kw": info["executed_power_kw"],
               "net_operating_cost": cost, "raw_reward_float32": raw_reward,
               "shaping_reward_usd": shaping_usd,
               "effective_unscaled_reward": effective_unscaled_reward,
               "scaled_reward_float32": scaled_reward,
               "value_prediction": value_prediction,
               "dollar_equivalent_value_prediction": value_prediction / self.reward_scale,
               "truncated": bool(info.get("TimeLimit.truncated", False)),
               "terminal_bootstrap_value": None,
               "dollar_equivalent_terminal_bootstrap_value": None,
               "power_balance_residual_kw": info["power_balance_residual_kw"],
               "energy_balance_residual_kwh": info["energy_balance_residual_kwh"]}
        if hasattr(distribution, "distribution") and hasattr(distribution.distribution, "probs"):
            row["categorical_probs"] = probs
            row["discrete_action"] = int(sampled)
        if row["truncated"]:
            row["episode_metrics"] = info["cumulative_metrics"]
            self.pending_terminal = row
        self.rows.append(row)
        # Persist raw transition evidence immediately, even if a rollout fails.
        self.emit("transitions", row)
        self.transition_count += 1
        return True

    def _on_rollout_end(self):
        if self.pending_terminal is not None:
            raise RuntimeError("SB3 did not compute the expected timeout bootstrap.")
        buffer = self.model.rollout_buffer
        if len(self.rows) != buffer.buffer_size:
            raise RuntimeError("Incomplete diagnostic rollout.")
        for i, row in enumerate(self.rows):
            row.update(value_target=float(buffer.returns[i, 0]),
                       advantage=float(buffer.advantages[i, 0]),
                       buffer_reward=float(buffer.rewards[i, 0]))
            row["dollar_equivalent_value_target"] = row["value_target"] / self.reward_scale
            row["dollar_equivalent_advantage"] = row["advantage"] / self.reward_scale
            row["dollar_equivalent_buffer_reward"] = row["buffer_reward"] / self.reward_scale
            expected = row["scaled_reward_float32"]
            if row["truncated"]:
                expected += self.model.gamma * row["terminal_bootstrap_value"]
            if not np.isclose(row["buffer_reward"], expected, rtol=1e-6, atol=1e-5):
                raise ValueError("Unexpected rollout reward/terminal bootstrap accounting.")
            self.emit("rollouts", row)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.parameter_snapshots = None
