"""LakeStack (NEW) — medallion lakehouse bucket + Glue database.

Per design Part C.2.3:
    - aws_s3.Bucket `amlas-lake-{account}-{region}` with bronze/silver/gold
      prefix convention (created implicitly by writers; we declare lifecycle rules
      keyed on those prefixes).
      - block public access, S3-managed encryption, enforce SSL, versioned.
      - lifecycle: expire `bronze/` objects after 30 days (cost control).
      - removal_policy=DESTROY + auto_delete_objects=True — this is a de-mock /
        test lake (OPEN DECISION 2 default), so per-run teardown is clean and the
        global S3 namespace is freed.
    - aws_glue.CfnDatabase `amlas_lake` (underscores allowed for Glue) when the
      Glue catalog backend is selected. Glue DB itself is $0 idle.
    - SSM params: lake-bucket, lake-silver-prefix, lake-gold-prefix, lake-bronze-prefix,
      glue-database → feed AGENTICML_LAKE_* env.

SEC-01: bucket name composed from `self.account`/`self.region` tokens only; no
hardcoded account IDs.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_glue as glue,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct

from amlas.config._env_compat import env_with_compat

from ._kms import make_data_key
from ._s3 import make_access_log_bucket
from ._tags import apply_cost_tags


class LakeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        region_name: str | None = None,
        glue_database_name: str | None = None,
        bronze_expiry_days: int = 30,
        kms_key_arn: str = "",
        prod: bool = False,
        # PERF-11: Athena partition projection on the gold table. ON by default (the gold
        # medallion table is date-partitioned + grows highly-partitioned), parameterized OFF
        # for SPARSE tables where projection can hurt (research §1.3 caveat). The projected
        # date key + storage-location template let Athena compute partition locations in-memory
        # from config instead of reading the Glue Catalog per query (lower planning latency).
        enable_partition_projection: bool = True,
        gold_table_name: str = "gold_features",
        partition_key: str = "dt",
        partition_projection_range: str = "2020-01-01,NOW",
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"
        # Default to the actual CDK stack region so direct callers are portable
        # across AWS regions; callers may still provide an explicit name when
        # intentionally targeting a different naming convention.
        self.region_name = region_name or self.region
        self.glue_database_name = (
            glue_database_name
            if glue_database_name is not None
            else f"{hyphen_project.replace('-', '_')}_lake"
        )
        self.enable_partition_projection = enable_partition_projection
        self.gold_table_name = gold_table_name
        self.partition_key = partition_key
        self.partition_projection_range = partition_projection_range
        # WA-21: in prod the lake bucket must RETAIN (no auto-delete) so a teardown can't
        # strand data; dev/test keeps the de-mock DESTROY+auto_delete for clean per-run
        # teardown. Driven by ``prod=`` (env ``AMLAS_ENV=prod`` → infra.app).
        self.prod = prod or env_with_compat("AMLAS_ENV", "").strip().lower() == "prod"
        # WA-12: stamp cost-allocation tags (project/env/owner) for chargeback.
        apply_cost_tags(self)

        # WA-01: customer-managed CMK (rotation on) — created by default, or the
        # adopter's shared/BYO key imported when ``kms_key_arn`` is supplied.
        self.data_key = make_data_key(
            self,
            "LakeDataKey",
            key_arn=kms_key_arn,
            alias=f"alias/{hyphen_project}-lake",
            description="Amlas lake CMK (rotation on) — medallion data at rest",
            enable_rotation=True,  # explicit at the call site (make_data_key default is also True)
        )

        self.access_log_bucket = make_access_log_bucket(self)
        self.lake_bucket = self._build_lake_bucket(bronze_expiry_days)
        self.glue_database = self._build_glue_database()

        self.silver_prefix = f"s3://{self.lake_bucket.bucket_name}/silver/"
        self.gold_prefix = f"s3://{self.lake_bucket.bucket_name}/gold/"
        self.bronze_prefix = f"s3://{self.lake_bucket.bucket_name}/bronze/"

        # PERF-11: a projected gold Glue table so Athena computes partition locations
        # in-memory from config (lower per-query planning latency on the date-partitioned
        # gold medallion table). Skipped when projection is disabled (sparse-table caveat).
        self.gold_table = self._build_gold_table() if self.enable_partition_projection else None

        self._write_ssm_params()
        self._emit_outputs()

    def _build_lake_bucket(self, bronze_expiry_days: int) -> s3.Bucket:
        return s3.Bucket(
            self,
            "LakeBucket",
            bucket_name=f"{self.hyphen_project}-lake-{self.account}-{self.region_name}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            bucket_key_enabled=True,
            versioned=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="lake-bucket/",
            # WA-21: RETAIN (no auto-delete) in prod; DESTROY+auto_delete in dev/test.
            removal_policy=RemovalPolicy.RETAIN if self.prod else RemovalPolicy.DESTROY,
            auto_delete_objects=not self.prod,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireBronzeRaw",
                    prefix="bronze/",
                    expiration=Duration.days(bronze_expiry_days),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
                s3.LifecycleRule(
                    id="CleanupOldVersions",
                    noncurrent_version_expiration=Duration.days(7),
                ),
            ],
        )

    def _build_glue_database(self) -> glue.CfnDatabase:
        return glue.CfnDatabase(
            self,
            "LakeGlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=self.glue_database_name,
                description="Amlas medallion lake catalog (bronze/silver/gold)",
            ),
        )

    def _build_gold_table(self) -> glue.CfnTable:
        """PERF-11: a Parquet gold Glue table with Athena PARTITION PROJECTION enabled.

        Partition projection has Athena compute partition locations IN-MEMORY from the
        ``projection.*`` table parameters instead of reading the Glue Catalog per query —
        cutting per-query planning latency on the highly-partitioned, date-keyed gold table
        (research §1.3). All values are parameterized (key / range / table name) so an adopter
        can tune them or disable projection entirely for a sparse table.

        The table is Parquet-backed (Snappy via SerDe defaults) over the ``gold/<table>/``
        prefix, with a single date partition key projected over ``partition_projection_range``
        and a ``storage.location.template`` so each partition's S3 path is computed, not listed.
        """
        location = f"{self.gold_prefix}{self.gold_table_name}/"
        proj = self.partition_key
        parameters = {
            "classification": "parquet",
            "parquet.compression": "SNAPPY",
            # PERF-11: partition projection — the planning-latency lever.
            "projection.enabled": "true",
            f"projection.{proj}.type": "date",
            f"projection.{proj}.range": self.partition_projection_range,
            f"projection.{proj}.format": "yyyy-MM-dd",
            f"projection.{proj}.interval": "1",
            f"projection.{proj}.interval.unit": "DAYS",
            # location template lets Athena compute each partition's S3 path in-memory.
            "storage.location.template": f"{location}${{{proj}}}/",
        }
        return glue.CfnTable(
            self,
            "LakeGoldTable",
            catalog_id=self.account,
            database_name=self.glue_database.ref,
            table_input=glue.CfnTable.TableInputProperty(
                name=self.gold_table_name,
                table_type="EXTERNAL_TABLE",
                parameters=parameters,
                partition_keys=[
                    glue.CfnTable.ColumnProperty(name=proj, type="string"),
                ],
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    location=location,
                    input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                    output_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                    serde_info=glue.CfnTable.SerdeInfoProperty(
                        serialization_library="org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                    ),
                ),
            ),
        )

    def _write_ssm_params(self) -> None:
        params = {
            "lake-bucket": self.lake_bucket.bucket_name,
            "lake-bronze-prefix": self.bronze_prefix,
            "lake-silver-prefix": self.silver_prefix,
            "lake-gold-prefix": self.gold_prefix,
            "glue-database": self.glue_database_name,
        }
        for key, value in params.items():
            ssm.StringParameter(
                self,
                f"Param{key.replace('-', ' ').title().replace(' ', '')}",
                parameter_name=f"{self.ssm_prefix}/{key}",
                string_value=value,
                description=f"Amlas lake output: {key}",
            )

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "LakeBucketName",
            value=self.lake_bucket.bucket_name,
            export_name=f"{self.hyphen_project}-lake-bucket",
        )
        CfnOutput(
            self,
            "LakeSilverPrefix",
            value=self.silver_prefix,
            export_name=f"{self.hyphen_project}-lake-silver-prefix",
        )
        CfnOutput(
            self,
            "LakeGoldPrefix",
            value=self.gold_prefix,
            export_name=f"{self.hyphen_project}-lake-gold-prefix",
        )
        CfnOutput(
            self,
            "LakeBronzePrefix",
            value=self.bronze_prefix,
            export_name=f"{self.hyphen_project}-lake-bronze-prefix",
        )
        CfnOutput(
            self,
            "GlueDatabaseName",
            value=self.glue_database_name,
            export_name=f"{self.hyphen_project}-glue-database",
        )
