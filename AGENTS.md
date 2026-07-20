# Hermes Auxiliary Brain - Development Guide

Keep this plugin small, boring, and useful. Done is better than perfect; the
release raccoon already has enough architecture diagrams.

## Design boundaries

- This is a standalone Hermes plugin. Do not edit or copy files into the
  `hermes-agent` checkout to implement plugin behavior.
- Reuse Hermes plugin APIs and existing auxiliary-brain modules. Prefer the
  smallest change that fixes the whole behavior.
- Keep Hermes' prompt prefix stable. Optional context belongs in the current
  turn through the existing hook.
- The normal plugin runtime stays dependency-free. ML packages belong only in
  the isolated profile-local training environments.
- Training, promotion, and rollback remain explicit local commands. Never
  train or activate an adapter automatically.
- Treat local-model output as untrusted data. It receives no tools or
  autonomous authority.

## State and privacy

Runtime and learned state live under `HERMES_HOME/auxiliary-brain/`, never in
this repository. Do not commit databases, exports, bundles, runs, logs,
adapters, credentials, or model caches.

When moving machines, preserve `brain.db` with a consistent SQLite backup and
rebuild platform-specific runtimes and environments. Follow the migration
section in `docs/training.md`.

## Validation

Use Python 3.11 or newer:

```console
python -m pip install pytest fastapi httpx ruff
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m compileall -q auxiliary_brain dashboard/plugin_api.py scripts __init__.py install.py
```

For integration against an adjacent Hermes checkout:

```console
HERMES_AGENT_ROOT=../hermes-agent python -m pytest tests/test_dashboard_auth_integration.py tests/test_hermes_integration.py -q
```

Do not download or load model weights merely to prove an unrelated code or
documentation change. Run heavyweight stages sequentially and record only
privacy-safe aggregate validation results.

## Releases

`auxiliary_brain/version.py` is canonical and must match `plugin.yaml`. Before
tagging `vMAJOR.MINOR.PATCH`, require a clean tree, the complete local checks,
and green GitHub CI. Tags are annotated and GitHub Releases describe the
operator-visible changes, safety boundaries, and remaining limitations.
