# Windows project storage setup

This setup moves project-managed caches, Python temporary files, Python
bytecode caches, experiment outputs, and optional new data storage away from
C:. It does not move the repository, existing datasets, checkpoints, or
virtual environment, and it does not change machine-wide Windows settings.
The checked-in VS Code paths are workspace-relative. Machine-specific drive
paths are stored only in the ignored `.env.storage` file generated locally.

## Findings and selected drive

Inspection of the current workspace found:

| Resource | Current location | Observed size/status |
|---|---|---|
| Repository | `E:\3.2\Adaptive Energy Brain\Adaptive-Energy-Brain` | Already on E: |
| Existing `.venv` | `<repository>\.venv` | Already on E:, 1.636 GB; Python 3.12.10, PyTorch 2.7.0+cpu; preserved |
| `python` from an unconfigured shell | `C:\Program Files\Python314\python.exe` | Global Python 3.14; VS Code and setup terminal now select the project Python 3.12 environment |
| Python 3.12 base interpreter | `C:\Users\<user>\AppData\Local\Programs\Python\Python312` | Required by the existing venv; not moved or reinstalled |
| pip download cache | `%LOCALAPPDATA%\pip\Cache` | About 0.212 GB on C: |
| PyTorch hub cache | `%USERPROFILE%\.cache\torch` | Present, 0 files |
| Hugging Face cache | `%USERPROFILE%\.cache\huggingface` | Present, 0 files; Hugging Face tooling is not used by this project |
| Matplotlib cache/config | `%USERPROFILE%\.matplotlib` | Present, 0 files |
| Windows temporary directory | `%LOCALAPPDATA%\Temp` | About 0.277 GB total; not attributed to this project and not deleted |
| Project datasets | `<repository>\data` | About 0.013 GB on E: |
| Project results | `<repository>\results` | About 0.007 GB on E: |
| Pytest cache | `<repository>\.pytest_cache` | On E:, negligible; the test command may disable it with `-p no:cacheprovider` |
| Git metadata | Repository-local `.git` | On E:; no separate Git temp/cache relocation was indicated |

At inspection, free space was **16.06 GB on C:**, **30.95 GB on E:**, and
**33.92 GB on F:**. Following the requested E:-first rule, the setup selected
`E:\AdaptiveEnergyBrainStorage` and requires at least 5 GB free. No project
files were found to be using C: as their primary storage. Python base
installations and the prior pip cache are the identified C:-resident
development resources.

The only measured, clearly disposable C: cache is the pip cache (about
**212 MB**). The 277 MB Windows Temp total is not counted as recoverable
project space because its contents may belong to unrelated applications.
No cleanup is performed by this setup.

## Apply the configuration

Open the repository folder in VS Code. The workspace defaults to the existing
Python interpreter at `.venv\Scripts\python.exe` and opens a PowerShell
profile that dot-sources the setup script. The script selects E: if it is
available with at least 5 GB free, otherwise F:, creates only missing
directories, applies variables to that terminal process, and activates the
existing external/project environment. It is safe to run again.

For a standalone PowerShell terminal, run from the project root:

```powershell
. .\scripts\setup_windows_storage.ps1
```

You can choose an available drive/root explicitly or point at an existing
external split bundle:

```powershell
. .\scripts\setup_windows_storage.ps1 -StorageRoot 'F:\AdaptiveEnergyBrainStorage'
. .\scripts\setup_windows_storage.ps1 -SplitRoot 'E:\AdaptiveEnergyBrainStorage\datasets\processed\splits_hypothetical_campus_v1'
```

`-StorageRoot` and `-SplitRoot` values are local choices; do not commit them.
If the script creates `.env.storage`, it remains ignored by the existing Git
ignore rules. The file is referenced by the workspace Python extension for
debug/run environments. The terminal startup profile applies the variables
directly so child processes inherit them. If you prefer a different
interpreter, use **Python: Select Interpreter** and choose the desired
environment.

## Storage layout and environment variables

The script creates the following under the selected root, without deleting or
moving existing content:

```text
E:\AdaptiveEnergyBrainStorage\
  venv\
  datasets\
  cache\misc\
  pip\
  torch\
  huggingface\
  matplotlib\
  temp\
  pycache\
  checkpoints\
  experiments\
  logs\
```

| Variable | Purpose/tool behavior |
|---|---|
| `PIP_CACHE_DIR` | pip's package/download cache; `python -m pip cache dir` verifies it |
| `TORCH_HOME` | PyTorch hub/model artifact cache; `torch.hub.get_dir()` reports its effective `hub` directory |
| `HF_HOME` | Hugging Face cache root if a later optional tool uses it; no Hugging Face dependency is installed |
| `MPLCONFIGDIR` | Matplotlib configuration and cache directory |
| `TMP`, `TEMP` | Temporary-file defaults used by Python `tempfile` and compatible Windows subprocesses |
| `PYTHONPYCACHEPREFIX` | Python's external `.pyc` cache prefix; source `__pycache__` directories are not modified |
| `COVERAGE_FILE` | Coverage.py data file location |
| `AEB_DATA_ROOT` | Optional external dataset root for future/manual data placement |
| `AEB_SPLIT_ROOT` | Existing or manually selected complete split bundle; no split is regenerated |
| `AEB_EXPERIMENTS_ROOT` | Forecasting experiment history and metadata root |
| `AEB_CHECKPOINTS_ROOT` | Forecasting checkpoint root, grouped by experiment ID |

The setup leaves the checked-in dataset and preprocessing artifacts in the
repository on E: by default. It sets `AEB_SPLIT_ROOT` to that existing
`data\processed\splits_hypothetical_campus_v1` bundle. To use a copied bundle
on E: or F:, pass its path with `-SplitRoot` or set `AEB_SPLIT_ROOT` in the
ignored `.env.storage` file, then restart the terminal.

The training CLI keeps the current relative config behavior when these
variables are absent. When set, `AEB_SPLIT_ROOT` overrides the model YAML
split-bundle path; `AEB_EXPERIMENTS_ROOT` places history/metadata externally;
and `AEB_CHECKPOINTS_ROOT` stores `best.pt` and `last.pt` under a matching
experiment subdirectory. Existing explicit CLI options take precedence.
Resume from the checkpoint path under `AEB_CHECKPOINTS_ROOT`; the CLI keeps
history in the experiment directory. Google Colab explicitly uses its Drive
paths and clears local Windows storage overrides before starting the CLI.

Pytest's `.pytest_cache` remains in the repository (already on E:). Python's
`tmp_path` files use the process temp directory, so new pytest temporary files
go to the selected root after a fresh configured terminal starts. Coverage
output uses `COVERAGE_FILE`. Git remains repository-local and is on E:.

## Virtual environment guidance

The existing `.venv` is already on E: and is about 1.636 GB. Its Python 3.12
and PyTorch/scientific imports were verified. Keep using it; do not copy it or
recreate it solely to change drives. VS Code is configured for the
workspace-relative `.venv\Scripts\python.exe`, and the PowerShell profile
activates it.

If a future clone has no `.venv`, the setup script suggests creating a *fresh*
environment under `<storage-root>\venv` using an already installed compatible
Python (the project targets Python 3.12). Install only the repository's
existing requirements in that new environment, test imports and the focused
suite, then select that interpreter. Do not copy a Windows virtual
environment. If an environment is on C:, preserve it until a new external
environment passes validation; this script never removes it.

## Verification commands

From a newly configured VS Code PowerShell terminal:

```powershell
python -c "import sys,tempfile; print(sys.executable); print(sys.prefix); print(tempfile.gettempdir()); print(sys.pycache_prefix)"
python -m pip cache dir
python -c "import torch; print(torch.__version__); print(torch.hub.get_dir()); print(torch.cuda.is_available())"
python -c "import os,matplotlib; print(os.environ['HF_HOME']); print(matplotlib.get_configdir())"
python -m pytest tests -q -p no:cacheprovider
```

The local PyTorch install is CPU-only; CUDA is expected to report unavailable.
The full CPU suite was run after the storage-path changes. The test process's
Python temporary directory and bytecode prefix resolve under the selected E:
storage root; the project's `.pytest_cache` remains on E:.

## Manual cache cleanup (not run)

The old pip cache can be cleared manually after review if you want to recover
its measured ~212 MB. This targets only pip's cache, not datasets, checkpoints,
source files, virtual environments, or other C: caches:

```powershell
$oldPipCache = Join-Path $env:LOCALAPPDATA 'pip\Cache'
.\.venv\Scripts\python.exe -m pip --cache-dir $oldPipCache cache purge
```

The script and tests do not run this command. Do not purge the global Windows
Temp folder or other users' caches as part of this project task. No files have
been moved or deleted. Existing datasets, results, checkpoints, and the
virtual environment remain in place on E:.
