# src/amlas/tools/lake

## Purpose

Resolve partitioned medallion datasets and retain the lake read/write boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Preserve tier/path/partition selection; the existing lake boundary rejects writes to the raw/bronze tier.
- `__init__.py:resolve_lake_bucket` — Rewrite a LOGICAL lake bucket alias to the resolved physical lake bucket.
- `__init__.py:parse_lake_uri` — Parse a medallion lake URI into a structured descriptor.
- `partition.py:partition_dataset` — Write a dataset as a Hive-style ``key=value/`` partition tree (DQ-LAKE-009).
- `partition.py:recommend_partition_strategy` — Recommend partition key(s) from a dataset's cardinality profile (DQ-LAKE-010).
- Owned files: `__init__.py`, `partition.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
