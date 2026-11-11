# PyTorch Environment Setup

## Supported environments

The project core dependencies remain in `requirements.txt`. ML development is
an explicit additional install; PyTorch Geometric, CUDA toolkits, RAG/LLM
packages, and orchestration frameworks are not required.

| Environment | Python | PyTorch wheel | Dependency file |
|---|---|---|---|
| Local development (verified here) | Python 3.12 on Windows | PyTorch 2.7.0 CPU-only | `requirements-ml.txt` |
| Google Colab GPU (optional) | Use the active Colab Python runtime | PyTorch 2.7.0 CUDA 12.8 | `requirements-ml-colab.txt` |

PyTorch 2.7.0 is the stable release selected from the official install
selector; it supports Python 3.12. The two version pins select separate official
CPU and CUDA 12.8 wheels. Check the
[official PyTorch selector](https://pytorch.org/get-started/locally/) for a
newer supported build before intentionally changing these pins.

## Local Windows setup

From the project root, activate the existing Python 3.12 environment and
install the project dependencies followed by the CPU-only PyTorch wheel:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements-ml.txt
python scripts/check_ml_environment.py
python -m pytest tests/test_ml_environment.py -q -p no:cacheprovider
```

The CPU wheel does not require NVIDIA drivers, CUDA, or a GPU. It is intended
for development and smoke tests, not GPU acceleration.

The verified local environment used Python 3.12.10 and PyTorch 2.7.0+cpu.
Relevant installed core packages were NumPy 2.5.4, pandas 3.1.1, Matplotlib
3.11.2, scikit-learn 1.9.1, PyYAML 6.0.3, and pytest 9.2.0. The installed
PyTorch runtime dependencies were filelock 4.2.10, typing-extensions 4.16.0,
SymPy 1.14.0, NetworkX 3.7, Jinja2 3.1.6, fsspec 2026.9.0, setuptools
84.2.1, MarkupSafe 3.0.4, and mpmath 1.3.0. These are an environment record;
the existing core `requirements.txt` is intentionally unchanged and does not
pin those project-wide package versions.

## Google Colab GPU setup

Select **Runtime → Change runtime type → GPU**. The project keeps a separately
pinned CUDA 12.8 wheel in `requirements-ml-colab.txt`. From the project root,
first check the Python version and driver/GPU provided by the selected runtime.
Colab commonly has the core scientific packages preinstalled; install the
project's base requirements only if one is missing. If the runtime supports
CUDA 12.8 and the wheel's Python version, install:

```python
!python scripts/check_ml_environment.py
%pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r requirements-ml-colab.txt
```

Restart the runtime if Colab requests it, then run:

```python
!python scripts/check_ml_environment.py
```

The environment check selects CUDA when `torch.cuda.is_available()` is true;
otherwise it selects CPU and still verifies a neural-network training step.
It reports the exact Python/PyTorch/CUDA runtime versions, selected device,
and CUDA device name/count. Colab images and assigned GPUs can change, so save
this report with each future experiment. If the hosted runtime cannot support
the pinned wheel, do not force-install an incompatible CUDA build: use the
runtime's compatible PyTorch build, record its exact version, and update the
Colab pin only after validation on that runtime.

For already provisioned Colab images, inspect the existing PyTorch runtime
first; avoid reinstalling it unless a reproducible version is required. This
repository task validates local CPU PyTorch only; a live Colab GPU was not
available for verification here.

## Device selection and graph dependencies

Training code should use the same automatic selection pattern as
`scripts/check_ml_environment.py`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

The current data contract supplies `edge_index` and node arrays. Standard
PyTorch tensor/indexing operations are sufficient for initial graph operations;
no PyTorch Geometric operators or data loaders are yet required. Reconsider
PyG only if a later architecture needs graph-specific operators that are
otherwise cumbersome or incorrect to implement.

## Verification script

`scripts/check_ml_environment.py` checks that PyTorch imports, performs CPU
tensor operations, completes a small network forward/backward/optimizer step,
and reports CUDA availability accurately. On a CUDA-enabled host, it also
executes the smoke step on the automatically selected CUDA device. It exits
nonzero with an explicit error if PyTorch is missing or a smoke test fails.
