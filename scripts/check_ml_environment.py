"""Report PyTorch capabilities and run a tiny CPU/available-device training step."""

import json
import platform
import sys


class MLEnvironmentError(RuntimeError):
    """The configured ML runtime is unavailable or failed its smoke test."""


def _training_smoke_test(torch, device):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    ).to(device)
    inputs = torch.randn(5, 4, device=device)
    targets = torch.randn(5, 1, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    gradients_finite = all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    if not gradients_finite:
        raise MLEnvironmentError(f"Non-finite or missing gradients on {device}.")
    optimizer.step()
    return {
        "device": str(device),
        "output_shape": list(predictions.shape),
        "loss": float(loss.detach().cpu().item()),
        "backward_pass": True,
        "finite_gradients": gradients_finite,
    }


def check_environment():
    """Return runtime versions and verify CPU plus automatically selected device."""
    try:
        import torch
    except ImportError as exc:
        raise MLEnvironmentError(
            "PyTorch is not installed. Install the ML requirements for this environment."
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    selected_device = torch.device("cuda" if cuda_available else "cpu")
    result = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "selected_device": str(selected_device),
        "cpu_smoke_test": _training_smoke_test(torch, torch.device("cpu")),
    }
    if cuda_available:
        result["cuda_device_count"] = torch.cuda.device_count()
        result["cuda_device_name"] = torch.cuda.get_device_name(0)
        result["selected_device_smoke_test"] = _training_smoke_test(
            torch, selected_device
        )
    else:
        result["cuda_device_count"] = 0
        result["cuda_device_name"] = None
        result["selected_device_smoke_test"] = result["cpu_smoke_test"]
    return result


def main():
    try:
        result = check_environment()
    except (MLEnvironmentError, RuntimeError) as exc:
        print(f"ML environment check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
