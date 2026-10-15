"""Security boundaries shared across AMLAS application layers."""

from .model_artifacts import (
    UntrustedArtifactError,
    dump_model_artifact,
    load_model_artifact,
)

__all__ = [
    "UntrustedArtifactError",
    "dump_model_artifact",
    "load_model_artifact",
]
