"""
Cleaner Module

This module reads the raw JSON created by Document Intelligence,
removes unwanted content, normalizes text, and saves a cleaned JSON file.

Current flow:

Raw JSON
   ↓
Remove page footers and page numbers
   ↓
Normalize whitespace
   ↓
Preserve useful headings, paragraphs, and tables
   ↓
Save cleaned JSON

This cleaned JSON will later be passed to the chunking step.
"""

import json
import re
from pathlib import Path
from typing import Any


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "raw"
)

CLEANED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned"
)


# =========================================================
# 2. Unwanted paragraph roles
# =========================================================

# These roles are useful for page layout,
# but they should not become part of retrieval chunks.
UNWANTED_ROLES = {
    "pageFooter",
    "pageHeader",
    "pageNumber",
    "ParagraphRole.PAGE_FOOTER",
    "ParagraphRole.PAGE_HEADER",
    "ParagraphRole.PAGE_NUMBER",
}


# =========================================================
# 3. Known repeated text patterns
# =========================================================

# This footer appears on every synthetic PDF page.
UNWANTED_TEXT_PATTERNS = [
    r"^SYNTHETIC TRAINING DATA - NOT A REAL BNY MELLON DOCUMENT$",
    r"^Page \d+$",
    r"^Page \d+ of \d+$",
]


# =========================================================
# 4. Normalize text
# =========================================================

def normalize_text(text: str) -> str:
    """
    Cleans spacing and line-break problems.

    Example:
        "Policy   version\\n is   active"

    Becomes:
        "Policy version is active"
    """

    if not text:
        return ""

    # Replace line breaks and tabs with spaces.
    cleaned_text = text.replace("\n", " ")
    cleaned_text = cleaned_text.replace("\t", " ")

    # Replace multiple spaces with one space.
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text,
    )

    return cleaned_text.strip()


# =========================================================
# 5. Check whether text should be removed
# =========================================================

def is_unwanted_text(text: str) -> bool:
    """
    Returns True when the text matches a known footer,
    page number, or unwanted repeated line.
    """

    normalized = normalize_text(text)

    for pattern in UNWANTED_TEXT_PATTERNS:
        if re.match(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):
            return True

    return False


# =========================================================
# 6. Clean page content
# =========================================================

def clean_pages(
    pages: list[dict],
) -> list[dict]:
    """
    Cleans page-level lines while preserving page numbers
    for citation support.

    We keep:
    - page number
    - useful lines
    - clean page text

    We remove:
    - footer text
    - page number text
    - empty lines
    """

    cleaned_pages = []

    for page in pages:
        cleaned_lines = []

        for line in page.get("lines", []):
            content = normalize_text(
                line.get("content", "")
            )

            if not content:
                continue

            if is_unwanted_text(content):
                continue

            cleaned_lines.append(
                {
                    "line_number": line.get(
                        "line_number"
                    ),
                    "content": content,
                }
            )

        page_text = " ".join(
            line["content"]
            for line in cleaned_lines
        )

        cleaned_pages.append(
            {
                "page_number": page.get(
                    "page_number"
                ),
                "line_count": len(
                    cleaned_lines
                ),
                "text": page_text,
                "lines": cleaned_lines,
            }
        )

    return cleaned_pages


# =========================================================
# 7. Clean paragraph content
# =========================================================

def clean_paragraphs(
    paragraphs: list[dict],
) -> list[dict]:
    """
    Removes page footers and page numbers from paragraphs.

    Section headings are preserved because they will be
    important for document-aware chunking.
    """

    cleaned_paragraphs = []

    for paragraph in paragraphs:
        role = paragraph.get(
            "role",
            "body",
        )

        content = normalize_text(
            paragraph.get(
                "content",
                "",
            )
        )

        if not content:
            continue

        if role in UNWANTED_ROLES:
            continue

        if is_unwanted_text(content):
            continue

        cleaned_paragraphs.append(
            {
                "paragraph_number": paragraph.get(
                    "paragraph_number"
                ),
                "role": role,
                "content": content,
            }
        )

    return cleaned_paragraphs


# =========================================================
# 8. Clean table content
# =========================================================

def clean_tables(
    tables: list[dict],
) -> list[dict]:
    """
    Cleans whitespace inside table cells.

    Tables remain separate because later we may create
    dedicated table chunks instead of mixing them blindly
    with normal paragraph text.
    """

    cleaned_tables = []

    for table in tables:
        cleaned_cells = []

        for cell in table.get(
            "cells",
            [],
        ):
            content = normalize_text(
                cell.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            cleaned_cells.append(
                {
                    "row_index": cell.get(
                        "row_index"
                    ),
                    "column_index": cell.get(
                        "column_index"
                    ),
                    "content": content,
                    "kind": cell.get(
                        "kind"
                    ),
                }
            )

        cleaned_tables.append(
            {
                "table_number": table.get(
                    "table_number"
                ),
                "row_count": table.get(
                    "row_count"
                ),
                "column_count": table.get(
                    "column_count"
                ),
                "cells": cleaned_cells,
            }
        )

    return cleaned_tables


# =========================================================
# 9. Build cleaned document object
# =========================================================

def build_cleaned_document(
    raw_data: dict[str, Any],
) -> dict:
    """
    Builds the cleaned document structure.

    The original raw JSON is never modified.
    """

    cleaned_pages = clean_pages(
        raw_data.get(
            "pages",
            [],
        )
    )

    cleaned_paragraphs = clean_paragraphs(
        raw_data.get(
            "paragraphs",
            [],
        )
    )

    cleaned_tables = clean_tables(
        raw_data.get(
            "tables",
            [],
        )
    )

    document_info = raw_data.get(
        "document_info",
        {},
    ).copy()

    document_info["processing_stage"] = (
        "cleaned"
    )

    document_info["cleaned_page_count"] = (
        len(cleaned_pages)
    )

    document_info[
        "cleaned_paragraph_count"
    ] = len(cleaned_paragraphs)

    document_info["cleaned_table_count"] = (
        len(cleaned_tables)
    )

    return {
        "document_info": document_info,
        "pages": cleaned_pages,
        "paragraphs": cleaned_paragraphs,
        "tables": cleaned_tables,
    }


# =========================================================
# 10. Load JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict:
    """
    Reads a JSON file and returns it as a Python dictionary.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw JSON file was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(
            json_file
        )


# =========================================================
# 11. Save cleaned JSON
# =========================================================

def save_cleaned_json(
    cleaned_data: dict,
    output_path: Path,
) -> None:
    """
    Saves the cleaned document as formatted JSON.
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
            cleaned_data,
            json_file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# 12. Clean one raw document
# =========================================================

def clean_document(
    input_path: str | Path,
) -> Path:
    """
    Reads one raw JSON file, cleans it,
    and saves the cleaned result.

    Returns:
        Path to the cleaned JSON file.
    """

    input_file = Path(
        input_path
    ).resolve()

    raw_data = load_json(
        input_file
    )

    cleaned_data = build_cleaned_document(
        raw_data
    )

    output_file = (
        CLEANED_OUTPUT_DIR
        / input_file.name
    )

    save_cleaned_json(
        cleaned_data=cleaned_data,
        output_path=output_file,
    )

    print(
        f"Raw paragraphs: "
        f"{raw_data['document_info']['paragraph_count']}"
    )

    print(
        f"Cleaned paragraphs: "
        f"{cleaned_data['document_info']['cleaned_paragraph_count']}"
    )

    print(
        f"Tables preserved: "
        f"{cleaned_data['document_info']['cleaned_table_count']}"
    )

    print(
        f"Cleaned JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 13. Run directly
# =========================================================

if __name__ == "__main__":
    sample_json = (
        RAW_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        output_path = clean_document(
            sample_json
        )

        print(
            "\nSuccess. Cleaned document is ready "
            "for document-aware chunking."
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
            f"\nCleaning failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected cleaning error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )