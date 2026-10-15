# src/amlas/cloud

## Purpose

Stage datasets/specifications, dispatch managed-runtime calls and resolve configured research secrets.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `dispatch.py:resolve_runtime_arn` — Entry-point resolver: explicit → env → SSM ``/amlas/agentcore-runtime-arn``.
- `dispatch.py:resolve_session_bucket` — Entry-point resolver: explicit → env → SSM ``/amlas/session-bucket`` (FIX-13).
- `sandbox_staging.py:build_write_file_code` — Return Python that writes ``content`` to ``target_path`` inside the sandbox.
- `sandbox_staging.py:stage_files_into_sandbox` — Write each ``{sandbox_target_path: local_source_path}`` into the sandbox.
- `secrets.py:resolve_tavily_key` — Fetch the Tavily key from Secrets Manager. Returns None if absent/empty.
- `secrets.py:resolve_secret_into_env` — Inject the Tavily key into ``os.environ[TAVILY_API_KEY]`` if available.
- Owned files: `__init__.py`, `dispatch.py`, `sandbox_staging.py`, `secrets.py`, `spec_builder.py`, `staging.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
