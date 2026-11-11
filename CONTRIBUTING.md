# Contributing

Contributions that improve evaluation correctness, reproducibility, accessibility, or the local MLX workflow are welcome.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities using the [Security Policy](.github/SECURITY.md).

## Development setup

```bash
uv sync --extra dev --extra dashboard --extra models
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Do not download model weights or run training in ordinary CI. Use the explicitly marked `finetuning-regression-tester demo` fixtures for offline interface work. Real MLX tests are hardware-dependent and must be opt-in.

## Changes

- Keep grading deterministic and preserve raw model outputs.
- Add cases for analysis or metric changes; do not infer results from unobserved model behavior.
- Keep synthetic training and held-out entities/templates split before generating examples.
- Document changed metric definitions or interpretation limits.
- Do not commit model weights, secrets, real user data, or generated experiment runs.
- Run Ruff and Pytest before opening a pull request.

Please include a concise description of behavior changes, test evidence, and any Apple Silicon checks that could not be run. See the [methodology](docs/methodology.md) before changing score semantics.
