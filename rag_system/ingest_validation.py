from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.loaders import find_pdfs, load_document_metadata


REQUIRED_METADATA_FIELDS = (
    "status",
    "version",
    "effective_date",
    "related_labs",
    "analysis_types",
)


@dataclass(frozen=True)
class IngestValidationReport:
    """Result of validating source documents before ingest."""

    document_count: int
    warnings: list[str]


class IngestValidationError(ValueError):
    """Raised when source documents are not safe to ingest."""


def validate_ingest_metadata(docs_dir: Path, metadata_dir: Path) -> IngestValidationReport:
    """Validate document metadata required for auditable production ingest."""
    pdfs = find_pdfs(docs_dir)
    errors: list[str] = []
    warnings: list[str] = []
    effective_codes: dict[str, list[str]] = defaultdict(list)

    for pdf in pdfs:
        metadata = load_document_metadata(pdf, metadata_dir)
        if not metadata:
            errors.append(
                f"{pdf.name}: missing metadata JSON. Create {metadata_dir / (pdf.stem + '.json')}."
            )
            continue

        missing_fields = [
            field for field in REQUIRED_METADATA_FIELDS if not has_metadata_value(metadata, field)
        ]
        if missing_fields:
            errors.append(f"{pdf.name}: missing required metadata: {', '.join(missing_fields)}.")

        for list_field in ("related_labs", "analysis_types"):
            if has_metadata_value(metadata, list_field) and not is_non_empty_string_list(
                metadata[list_field]
            ):
                errors.append(f"{pdf.name}: {list_field} must be a non-empty list of strings.")

        status = str(metadata.get("status", "")).strip()
        document_code = str(metadata.get("document_code", "")).strip()
        if status == "Effective" and document_code:
            effective_codes[document_code].append(pdf.name)

    for document_code, filenames in sorted(effective_codes.items()):
        if len(filenames) > 1:
            warnings.append(
                "Multiple Effective documents share "
                f"document_code={document_code}: {', '.join(sorted(filenames))}."
            )

    if errors:
        raise IngestValidationError("Ingest metadata validation failed:\n- " + "\n- ".join(errors))

    return IngestValidationReport(document_count=len(pdfs), warnings=warnings)


def has_metadata_value(metadata: dict[str, Any], field: str) -> bool:
    """Return whether metadata has a non-empty value for a field."""
    value = metadata.get(field)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def is_non_empty_string_list(value: Any) -> bool:
    """Return whether a value is a non-empty list of non-empty strings."""
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )
