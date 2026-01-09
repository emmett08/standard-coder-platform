"""Long-running, resumable training pipeline utilities.

This package provides:
- Pipeline configuration (TOML)
- Checkpointing utilities
- A resumable end-to-end training runner

The pipeline is designed to be robust to interruption: after restart,
run with --resume to continue from the last checkpoint.
"""
