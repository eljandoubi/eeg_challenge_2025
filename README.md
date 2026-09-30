# EEG Challenge — VQ-VAE Pretraining

[![CI](https://github.com/eljandoubi/eeg_challenge/actions/workflows/ci.yml/badge.svg)](https://github.com/eljandoubi/eeg_challenge/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A self-supervised **VQ-VAE pretraining pipeline for EEG signals**, built for the
[EEG Foundation Challenge](https://eeg2025.github.io/) on the HBN-EEG dataset. The
model uses [braindecode](https://braindecode.org/)'s `Labram` transformer as an
encoder/decoder backbone around a Gumbel-softmax vector quantizer, and trains at
scale with 🤗 [Accelerate](https://huggingface.co/docs/accelerate).

## Overview

* **Data**: EEG recordings are streamed via [`eegdash`](https://github.com/eegdash/eegdash)'s
  `EEGChallengeDataset` and cut into fixed-length windows (2s @ 100Hz = 200 samples)
  with `braindecode`.
* **Model** (`src/model.py`): a `Labram` encoder produces per-channel logits that
  are discretized by a learnable codebook (`VectorQuantizerEmbedding`, Gumbel-softmax
  with an annealed temperature), and a second `Labram` decodes the quantized
  codes back into the signal's amplitude and phase spectra.
* **Losses**: a Huber loss on the reconstructed FFT amplitude/phase, a codebook
  orthogonality loss, and an encoder-perplexity term, combined through
  self-adjusting weights (`VQVALossesWeights`) that rebalance after every
  evaluation pass.
* **Training** (`src/pretrainer.py`): a single, `accelerate`-native training loop
  with mixed-precision/multi-GPU/multi-node support, early stopping
  (`src/early_stop.py`), TensorBoard logging, and checkpoint resumption.

## Project layout

```text
.
├── Dockerfile              # CUDA-enabled image used for real (GPU) training
├── launch_script.sh        # Entrypoint: prepares the accelerate config, then launches training
├── requirements.txt        # Pinned runtime deps installed inside the Docker image
├── pyproject.toml          # Dev/test environment + tooling config (uv, pytest, ruff)
├── src/
│   ├── configs.py          # All dataclass configs (CLI-parsable via HfArgumentParser)
│   ├── data_utils.py       # Dataset download + windowing helpers (eegdash/braindecode)
│   ├── torch_dataset.py    # Lazy, LRU-cached torch Dataset over per-subject windows
│   ├── model.py            # VQ-VAE model (Labram encoder/decoder + vector quantizer)
│   ├── early_stop.py       # Accelerate-aware early stopping + checkpointing
│   ├── prepare_yaml.py     # Patches an accelerate config from env vars (rank/host/port)
│   ├── pretrainer.py       # Main training entrypoint
│   └── check_subjects.py   # Diagnostic script: subject overlap across releases
└── tests/                  # CPU-only unit tests (see Testing below)
```

## Requirements

* Python **3.13+**
* A **CUDA GPU** to actually pretrain the model at a useful scale (the provided
  `Dockerfile` is the supported way to do this).
* **No GPU is required** to install the project, run the linter, or run the
  unit test suite — everything under `tests/` runs on CPU.

## Installation

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                   # core + dev dependencies (pytest, ruff)
uv sync --extra notebook  # also pulls in jupyter/ipykernel for interactive exploration
```

This creates a `.venv/` with everything needed for development and testing.
Actual GPU training is done through the Docker image (see
[Docker / GPU training](#docker--gpu-training)), whose runtime dependencies are
pinned separately in `requirements.txt`.

## Usage

All configuration is expressed as dataclasses in `src/configs.py`
(`TrainingConfig`, `VQVAEConfig`, `DataConfig`, `VQVALossesWeights`,
`EarlyStoppingConfig`) and parsed from the command line with
🤗 Transformers' `HfArgumentParser`, so every field is available as a
`--field_name` (or `--field-name`) flag — run `python src/pretrainer.py --help`
for the full list.

### 1. Build the windowed datasets (`--process`, on by default)

The first invocation downloads/reads the raw recordings, cuts them into
stimulus-locked windows, and caches them to disk:

```bash
python src/pretrainer.py \
  --data_path ./data \
  --train_data ./work/train_data \
  --eval_data ./work/eval_data
```

By default, releases `R1-R11` except `R5` are used for training and `R5` is
held out for evaluation (`DataConfig.train_inter` / `eval_inter`). Pass
`--tasks passive|active|all|<task_name>` to restrict which HBN-EEG tasks are
included.

### 2. Train (`--no_process`)

Once the windowed datasets are cached, launch training (skipping the
processing step) with 🤗 Accelerate:

```bash
accelerate launch src/pretrainer.py \
  --no_process \
  --train_data ./work/train_data --nb_train_sub 23344 \
  --eval_data ./work/eval_data --nb_eval_sub 3326 \
  --train_batch_size 8 --lr 1e-4
```

Training logs to TensorBoard under `--project_dir` (default `./work`), saves a
checkpoint every `--eval_steps`, and stops early once the eval loss stops
improving for `--patience` evaluations (`EarlyStoppingConfig`).

## Docker / GPU training

The `Dockerfile` builds on `pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel` and
installs the exact versions pinned in `requirements.txt`. Its entrypoint
(`launch_script.sh`) does two things on every container start:

1. `src/prepare_yaml.py` patches a **base accelerate config** (pointed to by
   the `CONFIG_FILE` environment variable) with the current `RANK`,
   `MASTER_ADDR` and `MASTER_PORT` environment variables, and writes the
   result to `acc_cfg.yml`. This makes the same image usable across
   single-GPU, multi-GPU and multi-node (e.g. SLURM/Kubernetes) launches.
2. `accelerate launch --config_file acc_cfg.yml src/pretrainer.py "$@"` starts
   training, forwarding any extra arguments given to `docker run`.

```bash
docker build -t eeg-challenge .
docker run --gpus all \
  -e CONFIG_FILE=/configs/accelerate_base.yaml \
  -v /path/to/data:/data \
  -v /path/to/configs:/configs \
  eeg-challenge \
  --data_path /data --no_process --train_batch_size 32
```

## Testing

The test suite is entirely CPU-only and avoids any network access or real EEG
data: dataset/model classes are exercised with tiny synthetic tensors, and
I/O-heavy dependencies (`load_concat_dataset`, `EEGChallengeDataset`) are
mocked or simply never reached (validation errors short-circuit before any
network call).

```bash
uv run pytest                                      # run the whole suite
uv run pytest --cov=src --cov-report=term-missing  # with coverage
```

What's covered: config parsing/validation, the VQ-VAE forward/backward pass
(on a tiny model), early stopping + checkpointing, the accelerate-config
patching logic, task-name validation, and the subject/window indexing math in
`SubjectsDataset` (including its LRU cache). The full multi-GPU training loop
in `pretrainer.py` and the real dataset download path are intentionally left
untested here — they require a GPU and network/data access, which this suite
explicitly avoids.

## Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR:

* **lint** — `ruff check` and `ruff format --check`, executed via `uvx` in an
  isolated environment (no need to install the heavy ML dependencies just to
  lint).
* **test** — `uv sync --locked` followed by `uv run pytest`, on CPU.

## Development

* Formatting/linting: [`ruff`](https://docs.astral.sh/ruff/) (`uv run ruff check .`,
  `uv run ruff format .`).
* `src/` is a flat module layout (no package `__init__.py`); `pyproject.toml`
  sets `pythonpath = ["src"]` for pytest so modules import the same way both
  scripts (`python src/pretrainer.py`) and tests do.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
