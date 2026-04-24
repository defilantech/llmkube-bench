# Security Policy

## Reporting a vulnerability

If you believe you've found a security issue in this repo — a secret
accidentally committed to history, a credential leak in a manifest, a
misconfigured RBAC rule, or anything similar — please report it privately
rather than opening a public GitHub issue.

Email: **security@defilantech.com**

Include:
- A description of the issue
- The file and commit (if applicable)
- Steps to reproduce
- Any suggested fix

We'll acknowledge within 72 hours and coordinate a fix before public
disclosure.

## Scope

This repo is a reproducibility artifact (manifests, Python harness, shell
orchestrator) rather than a production service. The most likely security
issues are:

- Secrets committed to git (tokens, registry credentials, TLS keys)
- Overly permissive RBAC in the manifests
- Network configurations that expose inference pods unintentionally

Bugs in the benchmark methodology itself, suggestions for more accurate
measurements, and general issues should go in the public issue tracker
instead.
