# Zero-cost runner

Use one trusted Linux x64 machine as the GitHub self-hosted runner.

## Requirements

- Git
- `python3` 3.11+
- Python `venv` support

CI creates a persistent environment at `~/.cache/genesis-ai-ci/venv` and installs CPU PyTorch there only when missing.

Keep the runner process online while GitHub jobs should execute.
