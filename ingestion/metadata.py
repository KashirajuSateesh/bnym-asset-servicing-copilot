"""
Metadata Extraction Module

This module reads the cleaned document JSON and extracts
document-level metadata from the first metadata table.

Current flow:

Cleaned JSON
    ↓
Read document control table
    ↓
Extract policy metadata
    ↓
Validate important fields
    ↓
Save metadata-enriched JSON

The extracted metadata will later be attached to every chunk.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)

METADATA_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metadata"
)


# =========================================================
# 2. Metadata field mapping
# =========================================================

# The PDF table contains human-readable names such as
# "Policy ID" and "Effective Date".
#
# We convert them into consistent Python/JSON field names.
FIELD_MAPPING = {
    "Policy ID": "policy_id",
    "Version": "version",
    "Status": "status",
    "Effective Date": "effective_date",
    "Review / End Date": "end_date",
    "Supersedes": "supersedes",
    "Policy Owner": "policy_owner",
    "Business Unit": "business_unit",
    "Jurisdiction": "jurisdiction",
    "Classification": "classification",
    "Access Scope": "access_scope",
    "Synthetic Document": "synthetic_document",
}


# =========================================================
# 3. Required metadata fields
# =========================================================

# These fields are important for version-aware retrieval
# and security filtering.
REQUIRED_FIELDS = {
    "policy_id",
    "version",
    "status",
    "effective_date",
    "end_date",
    "business_unit",
    "classification",
    "access_scope",
}


# =========================================================
# 4. Load JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """
    Reads a JSON file and returns a Python dictionary.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cleaned JSON file was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


# =========================================================
# 5. Save JSON
# =========================================================

def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Saves metadata-enriched document JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            data,
            json_file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# 6. Rebuild table rows
# =========================================================

def rebuild_table_rows(
    table: dict[str, Any],
) -> list[list[str]]:
    """
    Converts individual Azure table cells into rows.

    Example input cells:

        row 0, column 0 -> Policy ID
        row 0, column 1 -> POL-SEC-001
        row 0, column 2 -> Version
        row 0, column 3 -> 3.0

    Becomes:

        [
            ["Policy ID", "POL-SEC-001", "Version", "3.0"]
        ]
    """

    row_count = table.get("row_count", 0)
    column_count = table.get("column_count", 0)

    # Create an empty table structure.
    rows = [
        ["" for _ in range(column_count)]
        for _ in range(row_count)
    ]

    for cell in table.get("cells", []):
        row_index = cell.get("row_index")
        column_index = cell.get("column_index")
        content = cell.get("content", "").strip()

        if row_index is None or column_index is None:
            continue

        if row_index >= row_count:
            continue

        if column_index >= column_count:
            continue

        rows[row_index][column_index] = content

    return rows


# =========================================================
# 7. Extract metadata from control table
# =========================================================

def extract_metadata_from_table(
    table: dict[str, Any],
) -> dict[str, str]:
    """
    Extracts key-value pairs from the first document
    control table.

    The table is expected to contain two key-value pairs
    per row:

        Policy ID | POL-SEC-001 | Version | 3.0
    """

    rows = rebuild_table_rows(table)

    extracted_metadata: dict[str, str] = {}

    for row in rows:
        # Process the first key-value pair.
        if len(row) >= 2:
            first_label = row[0].strip()
            first_value = row[1].strip()

            mapped_field = FIELD_MAPPING.get(
                first_label
            )

            if mapped_field and first_value:
                extracted_metadata[
                    mapped_field
                ] = first_value

        # Process the second key-value pair.
        if len(row) >= 4:
            second_label = row[2].strip()
            second_value = row[3].strip()

            mapped_field = FIELD_MAPPING.get(
                second_label
            )

            if mapped_field and second_value:
                extracted_metadata[
                    mapped_field
                ] = second_value

    return extracted_metadata


# =========================================================
# 8. Normalize metadata
# =========================================================

def normalize_metadata(
    metadata: dict[str, str],
) -> dict[str, Any]:
    """
    Standardizes metadata values.

    Examples:
    - status becomes uppercase
    - business unit becomes uppercase
    - synthetic_document becomes Boolean
    """

    normalized = metadata.copy()

    uppercase_fields = [
        "status",
        "business_unit",
        "jurisdiction",
        "classification",
        "access_scope",
    ]

    for field in uppercase_fields:
        value = normalized.get(field)

        if isinstance(value, str):
            normalized[field] = value.strip().upper()

    synthetic_value = normalized.get(
        "synthetic_document"
    )

    if isinstance(synthetic_value, str):
        normalized["synthetic_document"] = (
            synthetic_value.strip().upper()
            in {"YES", "Y", "TRUE"}
        )

    return normalized


# =========================================================
# 9. Validate date format
# =========================================================

def validate_date_field(
    field_name: str,
    value: str | None,
) -> None:
    """
    Validates that a date uses YYYY-MM-DD format.
    """

    if not value:
        return

    try:
        datetime.strptime(
            value,
            "%Y-%m-%d",
        )

    except ValueError as error:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format. "
            f"Received: {value}"
        ) from error


# =========================================================
# 10. Validate required metadata
# =========================================================

def validate_metadata(
    metadata: dict[str, Any],
) -> None:
    """
    Ensures required fields are available.

    This prevents documents with missing version or
    access-control metadata from moving further into
    the ingestion pipeline.
    """

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if not metadata.get(field)
    ]

    if missing_fields:
        raise ValueError(
            "Required metadata is missing: "
            + ", ".join(sorted(missing_fields))
        )

    validate_date_field(
        "effective_date",
        metadata.get("effective_date"),
    )

    validate_date_field(
        "end_date",
        metadata.get("end_date"),
    )


# =========================================================
# 11. Calculate current applicability
# =========================================================

def calculate_policy_applicability(
    metadata: dict[str, Any],
    business_date: date | None = None,
) -> dict[str, Any]:
    """
    Determines whether the policy version is applicable
    for a specific business date.

    We use:
    - policy status
    - effective date
    - end date

    We do not simply select the highest version number.
    """

    business_date = business_date or date.today()

    effective_date = datetime.strptime(
        metadata["effective_date"],
        "%Y-%m-%d",
    ).date()

    end_date = datetime.strptime(
        metadata["end_date"],
        "%Y-%m-%d",
    ).date()

    status = metadata["status"].upper()

    is_date_applicable = (
        effective_date
        <= business_date
        <= end_date
    )

    is_active_status = status == "ACTIVE"

    return {
        "business_date_used": business_date.isoformat(),
        "is_date_applicable": is_date_applicable,
        "is_active_status": is_active_status,
        "is_currently_applicable": (
            is_date_applicable
            and is_active_status
        ),
    }


# =========================================================
# 12. Build enriched document
# =========================================================

def build_metadata_document(
    cleaned_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Extracts metadata and adds it to the cleaned document.
    """

    tables = cleaned_data.get(
        "tables",
        [],
    )

    if not tables:
        raise ValueError(
            "No tables were found. "
            "Document metadata cannot be extracted."
        )

    # The first table in our policy PDFs contains
    # document-control metadata.
    metadata = extract_metadata_from_table(
        tables[0]
    )

    metadata = normalize_metadata(
        metadata
    )

    validate_metadata(
        metadata
    )

    applicability = calculate_policy_applicability(
        metadata
    )

    document_info = cleaned_data.get(
        "document_info",
        {},
    ).copy()

    document_info["processing_stage"] = (
        "metadata_enriched"
    )

    return {
        "document_info": document_info,
        "metadata": metadata,
        "applicability": applicability,
        "pages": cleaned_data.get(
            "pages",
            [],
        ),
        "paragraphs": cleaned_data.get(
            "paragraphs",
            [],
        ),
        "tables": cleaned_data.get(
            "tables",
            [],
        ),
    }


# =========================================================
# 13. Process one document
# =========================================================

def extract_document_metadata(
    input_path: str | Path,
) -> Path:
    """
    Reads cleaned JSON, extracts metadata, and saves
    metadata-enriched JSON.
    """

    input_file = Path(
        input_path
    ).resolve()

    cleaned_data = load_json(
        input_file
    )

    enriched_data = build_metadata_document(
        cleaned_data
    )

    output_file = (
        METADATA_OUTPUT_DIR
        / input_file.name
    )

    save_json(
        data=enriched_data,
        output_path=output_file,
    )

    metadata = enriched_data["metadata"]
    applicability = enriched_data[
        "applicability"
    ]

    print(
        f"Policy ID: {metadata['policy_id']}"
    )

    print(
        f"Version: {metadata['version']}"
    )

    print(
        f"Status: {metadata['status']}"
    )

    print(
        f"Effective date: "
        f"{metadata['effective_date']}"
    )

    print(
        f"End date: {metadata['end_date']}"
    )

    print(
        f"Business unit: "
        f"{metadata['business_unit']}"
    )

    print(
        f"Currently applicable: "
        f"{applicability['is_currently_applicable']}"
    )

    print(
        f"Metadata JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 14. Run directly
# =========================================================

if __name__ == "__main__":
    sample_json = (
        CLEANED_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        output_path = extract_document_metadata(
            sample_json
        )

        print(
            "\nSuccess. Metadata-enriched document "
            "is ready for chunking."
        )

        print(
            f"Output file: {output_path}"
        )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"\nMetadata extraction failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected metadata extraction error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )