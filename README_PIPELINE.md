# Resumable pipeline (Mac M-series friendly)

This repository contains a reference implementation of the *Standard Coder* (SCH) approach
(described in the paper in `standard_coder.pdf`) plus a pluggable Sprint Forecasting domain.

This document focuses on the **long-running pipeline runner** designed for laptops:
it provides a Rich progress UI and **checkpointing**, so if the process is interrupted
(or you close the lid and later restart), it can continue where it left off.

## 1) Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a config file:

```bash
standard-coder init-config pipeline_config.toml
# edit pipeline_config.toml
```

Run the pipeline:

```bash
standard-coder pipeline --config pipeline_config.toml
```

Stages (default): `mine,train_sch,forecast`

Run only a subset:

```bash
standard-coder pipeline --config pipeline_config.toml --stages mine,train_sch
```

## 2) What gets produced (outputs)

Under `[outputs].base_dir` the runner creates:

- `raw/`
  - `mined_commits.sqlite` commit metadata
  - `diffs/<repo>/...` cached patches (for feature extraction)
  - `blobs/<repo>/...` cached before/after blobs for Python (Upgrade A)
- `labels/datasets/`
  - `coding_train.npz` training dataset `(x, y)` for SCH model
- `features/change_rep/`
  - `change_rep.joblib` fitted change representation (tokens or hybrid)
- `models/`
  - `work_inference/<kind>/` per-author fitted curves (checkpointed)
  - `mdn/model.pt` or `quantile/model.pt`
- `checkpoints/`
  - `run_state.json` stage status
  - `mdn/checkpoint.pt` / `quantile/checkpoint.pt` epoch-level checkpoints
- `forecasts/`
  - `<sprint_id>_forecast.json` (if sprint inputs provided)
- `logs/run.log`
- `reports/` (reserved for evaluation artefacts)

## 3) Checkpointing and resume

- Mining is idempotent (SQLite upserts).
- Work inference is checkpointed per author (`models/work_inference/.../*.json`).
- Predictor training is checkpointed per epoch (`checkpoints/*/checkpoint.pt`).

On restart, the pipeline will:
- reuse cached diffs/blobs,
- reuse the dataset if present *and* a fitted change representation exists,
- resume predictor training from the last epoch checkpoint.

## 4) Apple Silicon optimisation (MPS)

Config defaults use `device = "mps"` for torch training.
If MPS isn't available for your environment, the runner falls back to CPU.

## 5) Forecasting inputs (optional)

To run forecasting you provide two JSON files:

- `examples/forecasting/sprint_inputs.example.json`
- `examples/forecasting/historical_outcomes.example.json`

Set them in `pipeline_config.toml`:

```toml
[forecasting]
enable = true
sprint_inputs_path = "examples/forecasting/sprint_inputs.example.json"
historical_outcomes_path = "examples/forecasting/historical_outcomes.example.json"
```


## 6) Multi-task (Upgrade C): coding + delivery effort

Set in `pipeline_config.toml`:

```toml
[training]
predictor_kind = "multitask"

[training.multitask]
enable = true
pull_requests_path = "examples/prs.example.json"
```

The PR file supplies coarse delivery signals (open→merge time, review rounds/comments, CI failures).
The implementation uses the platform’s multi-task model to learn:
- a **coding effort** distribution (SCH)
- a **delivery effort** proxy distribution (bounded, conservative)

This keeps SCH estimation as its own domain, while enabling downstream sprint forecasting.
