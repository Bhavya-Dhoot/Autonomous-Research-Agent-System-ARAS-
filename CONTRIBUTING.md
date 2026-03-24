# Contributing to ARAS

Thanks for your interest in improving ARAS.

## Getting started

- Fork the repository and create a feature branch.
- Set up a local environment using the instructions in `README.md`.
- Keep changes focused and include tests when behavior changes.

## Development workflow

- Run tests before opening a PR:

```bash
pytest -v
```

- Prefer clear commit messages describing intent.
- Update documentation when APIs, behavior, or configuration change.

## Security rule (non-negotiable)

- Do not commit secrets of any kind.
- Never include real values for API keys or tokens in code, tests, docs, logs, or screenshots.
- Use `.env.example` for placeholders and local `.env` for private values.

## Pull requests

- Explain what changed and why.
- Link related issues.
- Include validation notes (tests run, manual checks).

By contributing, you agree that your contributions are licensed under the repository `LICENSE`.
