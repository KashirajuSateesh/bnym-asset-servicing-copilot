"""
PII and Sensitive Data Processing Module

This module reads chunked document JSON, detects sensitive values,
masks them before embedding generation, and saves a PII-safe version.

Current pipeline:

Chunk JSON
    ↓
Detect sensitive values
    ↓
Apply masking rules
    ↓
Add PII audit metadata
    ↓
Save safe chunks
    ↓
Ready for embeddings

Important design decision:

We do not automatically remove every business identifier.

For example:
- Customer email should be masked.
- Phone number should be masked.
- SSN or tax identifier should be masked.
- Customer ID may be preserved because agents need it for operations.
- Account ID may be preserved because it is needed for MCP lookups.
- Trade ID may be preserved because it is a business reference.

This module is designed so those rules can be changed later.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


# =========================================================
# 1. Project paths
# =========================================================

# This file is located at:
# project_root/ingestion/pii_processing.py
#
# parent       -> ingestion/
# parent.parent -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input created by chunker.py
CHUNK_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
)

# Output used later by embedder.py
PII_SAFE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pii_safe"
)


# =========================================================
# 2. Masking configuration
# =========================================================

# These fields should be masked before embeddings,
# prompts, traces, logs, and semantic caching.
MASKING_RULES = {
    "email": {
        "replacement": "[EMAIL_MASKED]",
        "enabled": True,
    },
    "phone": {
        "replacement": "[PHONE_MASKED]",
        "enabled": True,
    },
    "ssn": {
        "replacement": "[SSN_MASKED]",
        "enabled": True,
    },
    "tax_id": {
        "replacement": "[TAX_ID_MASKED]",
        "enabled": True,
    },
    "credit_card": {
        "replacement": "[CARD_MASKED]",
        "enabled": True,
    },
    "ip_address": {
        "replacement": "[IP_MASKED]",
        "enabled": True,
    },
}


# =========================================================
# 3. Business identifiers
# =========================================================

# These identifiers are sensitive business references,
# but they may be required by the agents and MCP tools.
#
# Therefore, we detect and record them, but we do not mask
# them by default.
BUSINESS_IDENTIFIER_RULES = {
    "customer_id": {
        "pattern": r"\bCUST-\d{6}\b",
        "mask": False,
        "replacement": "[CUSTOMER_ID_MASKED]",
    },
    "account_id": {
        "pattern": r"\bACC-\d{7}\b",
        "mask": False,
        "replacement": "[ACCOUNT_ID_MASKED]",
    },
    "trade_id": {
        "pattern": r"\bTRD-\d{8}\b",
        "mask": False,
        "replacement": "[TRADE_ID_MASKED]",
    },
    "exception_id": {
        "pattern": r"\bEXC-\d{7}\b",
        "mask": False,
        "replacement": "[EXCEPTION_ID_MASKED]",
    },
    "break_id": {
        "pattern": r"\bBRK-\d{7}\b",
        "mask": False,
        "replacement": "[BREAK_ID_MASKED]",
    },
    "case_id": {
        "pattern": r"\bCASE-\d{7}\b",
        "mask": False,
        "replacement": "[CASE_ID_MASKED]",
    },
}


# =========================================================
# 4. Regex patterns for direct identifiers
# =========================================================

PII_PATTERNS = {
    # Matches common email formats.
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+"
        r"\.[A-Za-z]{2,}\b"
    ),

    # Matches common US phone formats:
    # 205-555-1234
    # (205) 555-1234
    # 205 555 1234
    # +1 205 555 1234
    "phone": re.compile(
        r"(?<!\d)"
        r"(?:\+?1[\s.-]?)?"
        r"(?:\(?\d{3}\)?[\s.-]?)"
        r"\d{3}[\s.-]?\d{4}"
        r"(?!\d)"
    ),

    # Matches SSN format:
    # 123-45-6789
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),

    # Matches masked or unmasked tax-style identifiers.
    #
    # Examples:
    # 12-3456789
    # ***-**-5625
    "tax_id": re.compile(
        r"\b\d{2}-\d{7}\b"
        r"|"
        r"\*{3}-\*{2}-\d{4}"
    ),

    # Matches basic card-number shapes.
    #
    # This is only format detection.
    # It does not confirm whether the number is valid.
    "credit_card": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),

    # Matches IPv4 addresses.
    "ip_address": re.compile(
        r"\b"
        r"(?:"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"\."
        r"){3}"
        r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"\b"
    ),
}


# =========================================================
# 5. Load JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """
    Reads chunk JSON from disk.

    Args:
        input_path:
            Path to the chunk JSON file.

    Returns:
        Parsed JSON as a Python dictionary.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunk JSON file was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


# =========================================================
# 6. Save JSON
# =========================================================

def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Saves PII-safe JSON to disk.
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
# 7. Detect direct PII
# =========================================================

def detect_direct_pii(
    text: str,
) -> dict[str, list[str]]:
    """
    Detects direct PII values using regex patterns.

    Returns a dictionary like:

    {
        "email": ["john@example.test"],
        "phone": ["205-555-1234"]
    }

    Duplicate values are removed.
    """

    findings: dict[str, list[str]] = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)

        # Remove duplicates while keeping values readable.
        unique_matches = sorted(
            set(matches)
        )

        if unique_matches:
            findings[pii_type] = unique_matches

    return findings


# =========================================================
# 8. Detect business identifiers
# =========================================================

def detect_business_identifiers(
    text: str,
) -> dict[str, list[str]]:
    """
    Detects operational identifiers such as:
    - customer IDs
    - account IDs
    - trade IDs
    - exception IDs

    These values are recorded for governance, but are not
    masked unless the related rule says mask=True.
    """

    findings: dict[str, list[str]] = {}

    for identifier_type, rule in (
        BUSINESS_IDENTIFIER_RULES.items()
    ):
        pattern = re.compile(
            rule["pattern"]
        )

        matches = pattern.findall(
            text
        )

        unique_matches = sorted(
            set(matches)
        )

        if unique_matches:
            findings[
                identifier_type
            ] = unique_matches

    return findings


# =========================================================
# 9. Mask direct PII
# =========================================================

def mask_direct_pii(
    text: str,
) -> tuple[str, dict[str, int]]:
    """
    Applies masking to direct PII.

    Returns:
        masked_text:
            Text after masking.

        mask_counts:
            Number of values masked by category.
    """

    masked_text = text
    mask_counts: dict[str, int] = {}

    for pii_type, pattern in PII_PATTERNS.items():
        rule = MASKING_RULES.get(
            pii_type,
            {},
        )

        if not rule.get(
            "enabled",
            False,
        ):
            continue

        replacement = rule.get(
            "replacement",
            "[MASKED]",
        )

        # subn returns:
        # 1. updated text
        # 2. number of replacements
        masked_text, replacement_count = (
            pattern.subn(
                replacement,
                masked_text,
            )
        )

        if replacement_count > 0:
            mask_counts[
                pii_type
            ] = replacement_count

    return masked_text, mask_counts


# =========================================================
# 10. Optionally mask business identifiers
# =========================================================

def process_business_identifiers(
    text: str,
) -> tuple[str, dict[str, int]]:
    """
    Masks business identifiers only when their rule says
    mask=True.

    Current default:
    Operational identifiers remain visible because agents
    need them for investigation and MCP tool calls.
    """

    processed_text = text
    mask_counts: dict[str, int] = {}

    for identifier_type, rule in (
        BUSINESS_IDENTIFIER_RULES.items()
    ):
        if not rule.get(
            "mask",
            False,
        ):
            continue

        pattern = re.compile(
            rule["pattern"]
        )

        replacement = rule.get(
            "replacement",
            "[BUSINESS_ID_MASKED]",
        )

        processed_text, replacement_count = (
            pattern.subn(
                replacement,
                processed_text,
            )
        )

        if replacement_count > 0:
            mask_counts[
                identifier_type
            ] = replacement_count

    return processed_text, mask_counts


# =========================================================
# 11. Process one chunk
# =========================================================

def process_chunk(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Detects and masks sensitive values in one chunk.

    The original content is not overwritten silently.
    We keep:
    - original character count
    - safe content
    - detected categories
    - masking counts
    - business identifiers found
    - whether the chunk is safe for embedding
    """

    processed_chunk = deepcopy(
        chunk
    )

    original_content = chunk.get(
        "content",
        "",
    )

    if not original_content:
        processed_chunk["pii_processing"] = {
            "pii_detected": False,
            "pii_types": [],
            "mask_counts": {},
            "business_identifiers_detected": {},
            "safe_for_embedding": True,
        }

        processed_chunk[
            "safe_content"
        ] = ""

        return processed_chunk

    # Detect PII before masking so we can record what was found.
    direct_pii_findings = detect_direct_pii(
        original_content
    )

    # Detect operational identifiers separately.
    business_identifier_findings = (
        detect_business_identifiers(
            original_content
        )
    )

    # Mask direct PII.
    safe_content, direct_mask_counts = (
        mask_direct_pii(
            original_content
        )
    )

    # Apply optional business-identifier masking.
    safe_content, business_mask_counts = (
        process_business_identifiers(
            safe_content
        )
    )

    all_mask_counts = {
        **direct_mask_counts,
        **business_mask_counts,
    }

    processed_chunk["safe_content"] = (
        safe_content
    )

    processed_chunk[
        "safe_character_count"
    ] = len(safe_content)

    processed_chunk["pii_processing"] = {
        "pii_detected": bool(
            direct_pii_findings
        ),
        "pii_types": sorted(
            direct_pii_findings.keys()
        ),

        # We do not store the actual detected direct PII values
        # in the processed file because that would create another
        # sensitive-data copy.
        "pii_value_count": sum(
            len(values)
            for values in direct_pii_findings.values()
        ),

        "mask_counts": all_mask_counts,

        # Operational identifiers may remain visible.
        # We record them because they affect access control
        # and tool routing.
        "business_identifiers_detected": (
            business_identifier_findings
        ),

        "business_identifiers_masked": (
            bool(business_mask_counts)
        ),

        "safe_for_embedding": True,
    }

    return processed_chunk


# =========================================================
# 12. Build the complete PII-safe document
# =========================================================

def build_pii_safe_document(
    chunk_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Processes every chunk and creates document-level
    PII statistics.
    """

    original_chunks = chunk_data.get(
        "chunks",
        [],
    )

    if not original_chunks:
        raise ValueError(
            "No chunks were found in the input document."
        )

    safe_chunks = []

    total_pii_chunks = 0
    total_masked_values = 0
    pii_type_counts: dict[str, int] = {}
    business_identifier_counts: dict[str, int] = {}

    for chunk in original_chunks:
        processed_chunk = process_chunk(
            chunk
        )

        safe_chunks.append(
            processed_chunk
        )

        pii_result = processed_chunk.get(
            "pii_processing",
            {},
        )

        if pii_result.get(
            "pii_detected"
        ):
            total_pii_chunks += 1

        for pii_type in pii_result.get(
            "pii_types",
            [],
        ):
            pii_type_counts[pii_type] = (
                pii_type_counts.get(
                    pii_type,
                    0,
                )
                + 1
            )

        total_masked_values += sum(
            pii_result.get(
                "mask_counts",
                {},
            ).values()
        )

        for identifier_type, values in (
            pii_result.get(
                "business_identifiers_detected",
                {},
            ).items()
        ):
            business_identifier_counts[
                identifier_type
            ] = (
                business_identifier_counts.get(
                    identifier_type,
                    0,
                )
                + len(values)
            )

    document_info = chunk_data.get(
        "document_info",
        {},
    ).copy()

    document_info["processing_stage"] = (
        "pii_safe"
    )

    return {
        "document_info": document_info,
        "metadata": chunk_data.get(
            "metadata",
            {},
        ),
        "applicability": chunk_data.get(
            "applicability",
            {},
        ),
        "chunk_summary": chunk_data.get(
            "chunk_summary",
            {},
        ),
        "pii_summary": {
            "total_chunks_processed": len(
                safe_chunks
            ),
            "chunks_with_direct_pii": total_pii_chunks,
            "total_values_masked": total_masked_values,
            "pii_type_counts": pii_type_counts,
            "business_identifier_counts": (
                business_identifier_counts
            ),
            "embedding_field": "safe_content",
            "raw_content_preserved": True,
        },
        "chunks": safe_chunks,
    }


# =========================================================
# 13. Process one chunk document
# =========================================================

def process_document_pii(
    input_path: str | Path,
) -> Path:
    """
    Reads chunk JSON, processes sensitive information,
    and saves the PII-safe JSON file.

    Returns:
        Path to the generated PII-safe file.
    """

    input_file = Path(
        input_path
    ).resolve()

    chunk_data = load_json(
        input_file
    )

    safe_data = build_pii_safe_document(
        chunk_data
    )

    output_file = (
        PII_SAFE_OUTPUT_DIR
        / input_file.name
    )

    save_json(
        data=safe_data,
        output_path=output_file,
    )

    pii_summary = safe_data[
        "pii_summary"
    ]

    print(
        f"Chunks processed: "
        f"{pii_summary['total_chunks_processed']}"
    )

    print(
        f"Chunks containing direct PII: "
        f"{pii_summary['chunks_with_direct_pii']}"
    )

    print(
        f"Total values masked: "
        f"{pii_summary['total_values_masked']}"
    )

    print(
        "Business identifiers detected: "
        f"{pii_summary['business_identifier_counts']}"
    )

    print(
        f"PII-safe JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 14. Run directly
# =========================================================

if __name__ == "__main__":
    sample_json = (
        CHUNK_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        output_path = process_document_pii(
            sample_json
        )

        print(
            "\nSuccess. PII-safe chunks are ready "
            "for embedding generation."
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
            f"\nPII processing failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected PII processing error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )