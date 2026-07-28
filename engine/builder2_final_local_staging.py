"""
Builder2 final-video local staging — caller-owned paths, safe handoff, no public URLs.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from engine.builder2_tournament_contracts import Builder2TournamentError


class Builder2FinalLocalStagingError(Builder2TournamentError):
    def __init__(self, code: str, *, stage: str = "local_staging") -> None:
        super().__init__(code)
        self.stage = stage


def is_legacy_headline_store_path(path: Path) -> bool:
    from engine.video_headline_postprocess import _storage_root

    try:
        root = _storage_root().resolve()
        resolved = path.resolve()
        return resolved.parent == root
    except OSError:
        return False


def handoff_local_final_artifact(source: Path, destination: Path) -> None:
    if is_legacy_headline_store_path(destination):
        raise Builder2FinalLocalStagingError(
            "builder2_final_legacy_headline_store_rejected",
            stage="local_staging",
        )
    if not source.is_file():
        raise Builder2FinalLocalStagingError(
            "builder2_final_local_source_missing",
            stage="local_staging",
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Builder2FinalLocalStagingError(
            "builder2_final_staging_parent_missing",
            stage="local_staging",
        ) from exc
    if not destination.parent.is_dir():
        raise Builder2FinalLocalStagingError(
            "builder2_final_staging_parent_missing",
            stage="local_staging",
        )
    try:
        if destination.exists():
            destination.unlink()
        shutil.copy2(source, destination)
    except FileNotFoundError as exc:
        if not source.is_file():
            raise Builder2FinalLocalStagingError(
                "builder2_final_local_source_missing",
                stage="local_staging",
            ) from exc
        raise Builder2FinalLocalStagingError(
            "builder2_final_staging_parent_missing",
            stage="local_staging",
        ) from exc
    except OSError as exc:
        raise Builder2FinalLocalStagingError(
            "builder2_final_local_handoff_failed",
            stage="local_staging",
        ) from exc
    if not destination.is_file():
        raise Builder2FinalLocalStagingError(
            "builder2_final_local_handoff_failed",
            stage="local_staging",
        )


def prepare_publication_staging(*, local_final_path: Path) -> dict[str, bool | int | None]:
    present = local_final_path.is_file()
    size_bytes = local_final_path.stat().st_size if present else None
    return {
        "publicationStagingPreparationAttempted": True,
        "publicationStagingPreparationAccepted": bool(present and size_bytes),
        "localFinalArtifactPresentAfterRender": bool(present),
        "localFinalArtifactSizeBytes": int(size_bytes) if size_bytes is not None else None,
        "legacyHeadlineStoreRejectedAsFinalDestination": True,
    }
