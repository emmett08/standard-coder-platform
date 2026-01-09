# Standard Coder Platform

This repository is a **reference implementation** of the paper
"The standard coder: a machine learning approach to measuring the effort
required to produce source code change" (Wright & Ziegler, 2019).

It implements:

- **Baseline SCH scoring pipeline (paper-like)**
  - Token features (bag-of-tokens)
  - Neural HMM work inference (coding vs not-coding)
  - Mixture Density Network (MDN) effort predictor

- **Upgrade A**
  - Structure-aware **AST edit features** (Python)
  - Simple **dependency/call graph** delta features (Python)

- **Upgrade B**
  - Hawkes-process based work inference
  - Variational / state-space style work inference (approximate EKF)

- **Upgrade C**
  - Multi-task outputs: coding effort (SCH) *and* delivery effort
  - Event timeline schema (commits, PR, CI, issue events)

- **Upgrade D**
  - Quantile regression head (pinball loss)
  - Optional dual-head predictor (MDN + quantiles)

- **Pluggable Sprint Forecasting**
  - Event-driven hooks (pub/sub)
  - Monte Carlo sprint completion probabilities
  - Scenario modelling (time off, scope growth, team changes)

> Note: This is an engineering-friendly scaffold that runs end-to-end on
> synthetic data and small repositories. For large-scale training you will
> want to batch data, use efficient time binning, and tune models.

## Quick start

Create a virtual environment and install:

```bash
pip install -e .[dev]
```

Run unit tests:

```bash
pytest
```

Run a toy end-to-end demo (simulated commits + training + forecast):

```bash
standard-coder demo
```

## CLI

```bash
standard-coder --help
```

## Design

The platform is split into two bounded contexts:

- `standard_coder.sch` – SCH estimation (change → effort distribution)
- `standard_coder.forecasting` – sprint forecasting (probabilistic planning)

They communicate via **domain events** in `standard_coder.integration`.

## Licence

This repository is provided as-is for evaluation and prototyping.
