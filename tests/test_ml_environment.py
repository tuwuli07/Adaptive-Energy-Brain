"""Runtime checks for the optional, locally pinned PyTorch environment."""

import json
from pathlib import Path
import subprocess
import sys

import torch

from scripts.check_ml_environment import check_environment


def test_cpu_tensor_operations_work():
    left = torch.tensor([1.0, 2.0], device="cpu")
    right = torch.tensor([3.0, 4.0], device="cpu")
    assert torch.equal(left + right, torch.tensor([4.0, 6.0]))


def test_cpu_neural_network_forward_and_backward_work():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 1))
    inputs = torch.randn(4, 3)
    targets = torch.randn(4, 1)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    assert loss.ndim == 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_environment_report_matches_torch_cuda_detection():
    result = check_environment()
    cuda_available = torch.cuda.is_available()
    assert result["torch_version"] == torch.__version__
    assert result["cuda_available"] is cuda_available
    assert result["selected_device"] == ("cuda" if cuda_available else "cpu")
    assert result["cpu_smoke_test"]["backward_pass"] is True
    assert result["cpu_smoke_test"]["finite_gradients"] is True
    assert result["selected_device_smoke_test"]["backward_pass"] is True
    assert result["cuda_device_count"] == (
        torch.cuda.device_count() if cuda_available else 0
    )
    assert result["cuda_device_name"] == (
        torch.cuda.get_device_name(0) if cuda_available else None
    )


def test_environment_script_reports_json():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_ml_environment.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["torch_version"] == torch.__version__
    assert report["cuda_available"] is torch.cuda.is_available()
