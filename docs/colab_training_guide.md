# Part 5.9 — Google Colab GPU training guide

The notebook at
[train_gnn_transformer_colab.ipynb](../notebooks/train_gnn_transformer_colab.ipynb)
is a runner for the repository's existing Part 5.8 training code. It does not
reimplement the model, preprocessing, or training loop. It trains from train
and validation partitions only; the held-out test data is not loaded or used.
The available split bundle is synthetic and hypothetical, not measured campus
data.

## Prepare the repository and open Colab

1. If this repository is not already pushed to GitHub, publish it yourself
   using your normal review and approval process. This project session did not
   commit or push anything. Include the notebook and training code at the same
   commit. Do not put credentials in the repository or notebook.
2. Open the notebook in Colab:
   - After the notebook is pushed, open the repository's notebook URL in
     Colab, or navigate to the repository in GitHub and use **Open in
     Colab**.
   - Before it is pushed, download or open the local
     `notebooks/train_gnn_transformer_colab.ipynb` and upload it using Colab's
     **File → Upload notebook**.
3. Select **Runtime → Change runtime type → GPU**, save, and reconnect.
4. In the repository-setup cell, set `REPOSITORY_URL` to the HTTPS Git remote.
   Choose a branch/ref with `REPOSITORY_REF`, or pin `COMMIT_SHA` to a full
   commit. The notebook records and validates the actual checked-out commit.
   For a private repository, store a token in Colab **Secrets** as
   `GITHUB_TOKEN`, set `USE_PRIVATE_REPO = True`, and grant this notebook
   access. Do not embed tokens in URLs or cells.

## Google Drive data setup

The split loader needs the saved Part 2 artifacts; it does not regenerate
splits or refit scalers. From the local project, copy the complete folder
`data/processed/splits_hypothetical_campus_v1` to:

```text
MyDrive/adaptive-energy-brain/data/processed/splits_hypothetical_campus_v1/
```

Preserve the `train/`, `validation/`, and `test/` directories and the
`split_manifest.json` and `preprocessing.json` files. The loader reads only
train and validation records; the test files remain unused. The manifest
contains boundary/checksum metadata, which is not a substitute for opening or
evaluating test data.

The notebook defines configurable paths for the Drive project/data root,
split bundle, preprocessing artifact, repository graph/sequence config, and
training outputs. Change the path variables if you use another Drive
directory. It reports a clear missing-artifact error rather than silently
generating or replacing data. Dataset CSVs and checkpoints are large
artifacts; keep them out of Git unless the project has explicitly chosen
otherwise.

The checked-in configuration expects 24-hour history and horizon, eight nodes
in the order `B01, B02, B03, B04, B05, PV01, W01, BUS01`, and seven kW targets
(`B01`–`B05`, `PV01`, `W01`). The assumed shared-bus graph has 14 directed
edges and no measured edge weights. `BUS01` is a context node with a false
target mask. Historical global weather (`global_X`) is prepared by the dataset
but is not passed to the current model. No known future covariates are used.

## Runtime and dependency policy

Run the GPU runtime check before dependency installation. It prints Python,
PyTorch, CUDA runtime, GPU name/count, and tests CUDA tensor and neural-network
backward operations. If CUDA is unavailable, stop, select a GPU runtime, and
rerun; full training must not fall back to CPU.

The notebook installs the project's base `requirements.txt`, which does not
replace PyTorch. It keeps Colab's existing PyTorch build when that build
passes the CUDA check. It intentionally does not auto-install
`requirements-ml-colab.txt` (`torch==2.7.0+cu128`): Colab images, Python
versions, and GPU drivers change, and an incompatible wheel can break the
runtime. Only consider that optional pin after checking Python compatibility
and the active runtime's PyTorch/CUDA support. Record the actual versions in
the saved experiment metadata.

## Smoke test, training, and recovery

Run the notebook top-to-bottom through the GPU smoke test. Its data preflight
uses `load_forecasting_train_validation_dataloaders`, validates saved
preprocessing, chronological train/validation boundaries, graph and target
ordering, shapes, kW units, and the `BUS01` mask. The GPU smoke test performs
one actual-data forward/backward pass and AdamW update using
`masked_mse_loss`; BUS01 is excluded from that loss.

Review `configs/forecasting_training.yaml` before running the full-training
cell. Its defaults are batch size 32, AdamW, learning rate 0.001, weight decay
0.0001, up to 50 epochs, patience 8, min-delta 0.0001, and seed 42. The
training CLI writes the best validation checkpoint and latest resumable
checkpoint after every completed epoch. Outputs are persisted directly under:

```text
MyDrive/adaptive-energy-brain/results/forecasting/<experiment-id>/
```

This directory contains `best.pt`, `last.pt`, `training_history.json`,
`experiment.json`, and (after the plot cell) `training_history.png`. Metadata
records the actual Git commit SHA, PyTorch/software versions, CUDA runtime and
GPU details, configs, split and preprocessing hashes, graph fingerprint,
target ordering/units, validation-selected best epoch/loss, and duration.
Checkpoint model weights/config do not embed graph edges or preprocessing
scalers; retain the matching repository configs and split bundle.

If Colab disconnects, reconnect, rerun setup/preflight, and set
`RESUME_FROM` in the training cell to the same experiment's Drive
`last.pt`. The cell derives the existing experiment ID from that checkpoint
and the CLI restores optimizer, early-stopping, and RNG state. Do not change
the experiment's data, graph, preprocessing, seed, or other training settings
when resuming. Keep the same `max_epochs` or increase it; the trainer rejects
incompatible resumes. Checkpoint writes occur once per epoch, so work since
the last completed checkpoint may need to be repeated.

The loss plot shows only recorded train and validation masked MSE. The reload
check reconstructs the model from `best.pt` and performs inference on a
validation batch; it verifies finite output, target order, shape, and the zero
BUS01 placeholder. Do not run final test evaluation in Part 5.9.

## Verification boundary

At the time this guide's local-readiness checklist was prepared, this Windows
workspace had CPU-only PyTorch and no live Colab runtime; GPU training was
therefore unverified at that point. The later user-reported Colab run completed
training and final evaluation. Its reported outcomes and external artifact
locations are documented in the [Part 5 completion report](part5_completion_report.md).
The Drive files are not assumed to be available in this local checkout.

Part 5.9 itself still does not load or evaluate the held-out test split;
Part 5.10 is reported complete separately. Do not use the test results for
further model selection or tuning.

## Beginner-friendly execution checklist

The following paths and readiness facts were verified in an earlier Windows
workspace snapshot. That snapshot predates the reported GitHub/Drive/Colab
run; it is retained as historical setup context, not current execution status:

- Local project: `E:\3.2\Adaptive Energy Brain\Adaptive-Energy-Brain`
- Dataset bundle: `E:\3.2\Adaptive Energy Brain\Adaptive-Energy-Brain\data\processed\splits_hypothetical_campus_v1`
- Drive bundle destination:
  `My Drive/adaptive-energy-brain/data/processed/splits_hypothetical_campus_v1`
- Drive experiment destination:
  `My Drive/adaptive-energy-brain/results/forecasting/<experiment-id>`
- The bundle is 7,324,231 bytes with 32 files. Its manifest has 31 SHA-256
  entries, including 10 for the test partition. The test data is included for
  later Part 5.10 evaluation; Part 5.9 does not open it.

### A. Prepare and publish a reviewed source revision

1. In VS Code, open the local project folder above. Open **Terminal → New
   Terminal** and run:

   ```powershell
   git status -sb
   git remote -v
   git log -1 --oneline
   ```

   At the time of this check, the repository had no commits and no remote;
   project files were untracked. If your output differs, stop and inspect your
   current branch and changes before proceeding.
2. In a browser, sign in to GitHub and create or select the intended
   repository. If creating one for this local project, do not initialize it
   with a README, license, or template; the local tree already contains the
   project. Copy the HTTPS repository URL. Do not send credentials to the
   terminal or store a token in the notebook.
3. Before staging, inspect `git status --short --ignored`. The ignored
   `.env.storage`, `.venv`, processed datasets, generated results, and model
   artifacts must stay local. `.env.example` is the safe template; never stage
   a real `.env` file or API key. Stage only reviewed source, configuration,
   tests, and documentation. Then inspect `git diff --cached --stat` and
   `git status --short` to ensure no datasets, checkpoints, or secrets are
   staged.
4. No commit or push was made in this task. Only after you explicitly approve
   publishing the reviewed tree, configure your actual URL and publish it:

   ```powershell
   git remote add origin https://github.com/<YOUR-ACCOUNT>/<YOUR-REPOSITORY>.git
   git branch -M main
   git add AGENTS.md .env.example .gitignore .vscode README.md configs dashboard docs main.py models notebooks requirements-ml-colab.txt requirements-ml.txt requirements.txt scripts src tests
   git add data\raw\.gitkeep data\processed\.gitkeep models\.gitkeep results\.gitkeep
   git diff --cached --stat
   git status --short
   git commit -m "Prepare Adaptive Energy Brain for Colab training" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
   git push -u origin main
   ```

   Replace the placeholder URL. If `origin` already exists, inspect it and
   use `git remote set-url origin <HTTPS-URL>` only if it is the intended
   remote. Confirm the push succeeded and record the full commit SHA with
   `git rev-parse HEAD`. Do not add `data/processed` split contents or
   `results`/`models` outputs to Git.

### B. Upload the exact dataset bundle to Drive

1. In Windows File Explorer, open the local bundle path above. Confirm it
   contains `train`, `validation`, `test`, `preprocessing.json`, and
   `split_manifest.json`.
2. In a browser, open Google Drive → **My Drive**. Create the parent folders
   `adaptive-energy-brain`, then `data`, then `processed` if absent.
3. Select **New → Folder upload** and choose the local
   `splits_hypothetical_campus_v1` folder. Wait for Drive's upload indicator
   to finish. The destination must be
   `My Drive/adaptive-energy-brain/data/processed/splits_hypothetical_campus_v1/`.
   Do not upload the whole project or create a second copy elsewhere.
4. In Drive, open the uploaded folder and confirm the three split directories
   and two JSON files are present. The notebook preflight will require a
   checksummed test partition directory, but will report that it was not
   loaded for training. If a folder is missing, upload the complete bundle
   again to the exact parent path; do not regenerate or edit the splits.

### C. Run the notebook in Colab

1. After the approved source commit is on GitHub, open
   `notebooks/train_gnn_transformer_colab.ipynb` in the repository and choose
   **Open in Colab**. Alternatively, in Colab choose **File → Upload
   notebook** and select that local notebook; the source commit still must be
   available at the configured GitHub URL because the notebook clones the
   repository to run its modules.
2. In Colab choose **Runtime → Change runtime type → T4 GPU** (or another
   available GPU), then **Save**. In the first GPU verification cell, expect
   CUDA available, a GPU name, and successful tensor and model
   forward/backward checks. If CUDA is unavailable, stop, change the runtime
   type, reconnect, and rerun from the first cell. Do not proceed on CPU.
3. Run the cells in order. In the repository setup cell, set
   `REPOSITORY_URL` to the exact HTTPS URL and `REPOSITORY_REF` to the
   published branch, or set `COMMIT_SHA` to the full 40-character SHA from
   the approved revision. For a private repository only, store the token in
   Colab **Secrets** as `GITHUB_TOKEN`, enable notebook access, and set
   `USE_PRIVATE_REPO = True`.
4. Allow the dependency cell to install base requirements. It should retain
   Colab's existing PyTorch build after the CUDA check; do not manually
   replace PyTorch with the separately pinned wheel unless compatibility has
   been checked. The Drive cell prompts you to authorize mounting Drive.
5. Before full training, verify the displayed bundle path is
   `/content/drive/MyDrive/adaptive-energy-brain/data/processed/splits_hypothetical_campus_v1`.
   Preflight should confirm train, validation, test-directory presence,
   train-fitted scaling, graph/target compatibility, and test checksums. The
   loader must report only train and validation records. The GPU smoke test
   should print a successful CUDA forward/backward/optimizer step, finite
   masked loss, expected `[batch, 24, 8, 1]` output, and BUS01 excluded.
   If any check fails, stop and resolve the reported path/schema/compatibility
   issue before training; do not proceed by bypassing validation.
6. Review the training configuration cell. It must show batch size 32,
   learning rate 0.001, weight decay 0.0001, maximum 50 epochs, patience 8,
   seed 42, and CUDA. Run the full-training cell once. Expect epoch-by-epoch
   train/validation masked-loss output and checkpoint updates under
   `My Drive/adaptive-energy-brain/results/forecasting/<experiment-id>/`.
   Full training is not a two-epoch smoke run; the metadata/history must show
   use of the complete train and validation partitions and no test use.
7. Wait for training, history plotting, Drive artifact verification, and
   `best.pt` validation reload cells to finish before disconnecting. In Drive,
   confirm `best.pt`, `last.pt`, `experiment.json`,
   `training_history.json`, and `training_history.png` exist. Record the
   experiment ID, actual Git SHA, GPU/CUDA details, completed/best epoch,
   best validation loss, duration, and reload result.

If Colab disconnects during training, reconnect with a GPU, rerun setup and
preflight, and set `RESUME_FROM` to that same experiment's Drive `last.pt`.
Use the same split, preprocessing, graph, and settings. Do not start a new
experiment using the prior experiment ID, and do not run the final test
evaluation until the completed run and reload evidence have been reviewed.
