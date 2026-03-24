# Security Policy

## Supported scope

This project is maintained on a best-effort basis for open-source use.

## Reporting a vulnerability

Please report vulnerabilities privately by contacting the maintainer before public disclosure.

When reporting, include:

- clear reproduction steps
- affected file/module
- impact assessment
- any suggested fix

## Secret handling policy

This repository enforces a strict no-secrets rule:

- Never commit API keys, tokens, passwords, certificates, or private keys.
- Store local credentials only in `.env` files that are git-ignored.
- Keep placeholder-only values in `.env.example`.
- Rotate credentials immediately if exposure is suspected.

## Hardening checklist for maintainers

- Enable GitHub secret scanning and push protection
- Require pull request reviews for default branch
- Use branch protection rules
- Rotate tokens regularly and use least privilege scopes
