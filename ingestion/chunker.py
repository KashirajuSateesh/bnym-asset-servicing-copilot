"""
Document-Aware Chunking Module

This module reads the metadata-enriched document JSON and creates
smaller retrieval chunks.

Current pipeline:

Metadata-enriched JSON
        ↓
Detect section headings
        ↓
Group paragraphs under sections
        ↓
Split long sections safely
        ↓
Create separate table chunks
        ↓
Attach policy and security metadata
        ↓
Save chunk JSON

These chunks will later be:
- masked for sensitive information
- converted into embeddings
- indexed into Azure AI Search
"""

import json
import re
from pathlib import Path
from typing import Any


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metadata"
)

CHUNK_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
)


# =========================================================
# 2. Chunk configuration
# =========================================================

# Maximum number of characters allowed in one chunk.
#
# This is a simple starting value.
# Later, we can replace character counting with token counting.
MAX_CHUNK_CHARACTERS = 800

# Minimum useful size for a chunk.
#
# Very small text blocks are usually less useful for retrieval.
MIN_CHUNK_CHARACTERS = 100

# Number of characters copied from the previous chunk.
#
# Overlap helps preserve context when one section must be split.
CHUNK_OVERLAP_CHARACTERS = 150


# =========================================================
# 3. Paragraph role values
# =========================================================

SECTION_HEADING_ROLES = {
    "sectionHeading",
    "ParagraphRole.SECTION_HEADING",
}

TITLE_ROLES = {
    "title",
    "ParagraphRole.TITLE",
}


# =========================================================
# 4. Load JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """
    Reads a JSON file and returns it as a Python dictionary.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Metadata JSON file was not found: {input_path}"
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
    Saves chunk data as formatted JSON.
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
# 6. Normalize text
# =========================================================

def normalize_text(
    text: str,
) -> str:
    """
    Removes unnecessary spaces and line breaks.
    """

    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# 7. Detect page number for text
# =========================================================

def find_page_number_for_text(
    text: str,
    pages: list[dict[str, Any]],
) -> int | None:
    """
    Attempts to find which page contains the paragraph text.

    This helps future citations include the source page.

    Since Document Intelligence paragraphs currently do not
    contain page references in our simplified JSON, we compare
    paragraph text against each page's extracted text.
    """

    normalized_text = normalize_text(text)

    if not normalized_text:
        return None

    # Use a smaller portion of the paragraph for matching.
    search_text = normalized_text[:150]

    for page in pages:
        page_text = normalize_text(
            page.get(
                "text",
                "",
            )
        )

        if search_text in page_text:
            return page.get(
                "page_number"
            )

    return None


# =========================================================
# 8. Group paragraphs by section
# =========================================================

def group_paragraphs_by_section(
    paragraphs: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Groups body paragraphs under their closest section heading.

    Example:

        3. Sensitive Data Handling
        Mask direct identifiers...
        Store only approved fields...

    Becomes one section object:

        {
            "section_title": "3. Sensitive Data Handling",
            "paragraphs": [...]
        }
    """

    sections: list[dict[str, Any]] = []

    current_section_title = "Document Introduction"
    current_paragraphs: list[dict[str, Any]] = []

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

        # Title becomes the first logical section.
        if role in TITLE_ROLES:
            if current_paragraphs:
                sections.append(
                    {
                        "section_title": current_section_title,
                        "paragraphs": current_paragraphs,
                    }
                )

            current_section_title = content
            current_paragraphs = []
            continue

        # When a new heading is found, close the previous section.
        if role in SECTION_HEADING_ROLES:
            if current_paragraphs:
                sections.append(
                    {
                        "section_title": current_section_title,
                        "paragraphs": current_paragraphs,
                    }
                )

            current_section_title = content
            current_paragraphs = []
            continue

        page_number = find_page_number_for_text(
            text=content,
            pages=pages,
        )

        current_paragraphs.append(
            {
                "content": content,
                "page_number": page_number,
                "paragraph_number": paragraph.get(
                    "paragraph_number"
                ),
            }
        )

    # Add the final section after the loop finishes.
    if current_paragraphs:
        sections.append(
            {
                "section_title": current_section_title,
                "paragraphs": current_paragraphs,
            }
        )

    return sections


# =========================================================
# 9. Split long text safely
# =========================================================

def split_long_text(
    text: str,
    max_characters: int = MAX_CHUNK_CHARACTERS,
    overlap_characters: int = CHUNK_OVERLAP_CHARACTERS,
) -> list[str]:
    """
    Splits long text into smaller parts.

    The function tries to split at sentence boundaries instead
    of cutting words in the middle.

    Overlap is added so context from the previous part is not lost.
    """

    text = normalize_text(text)

    if len(text) <= max_characters:
        return [text]

    # Split text into sentences.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            f"{current_chunk} {sentence}".strip()
        )

        if len(candidate) <= max_characters:
            current_chunk = candidate
            continue

        # Save the current chunk before starting a new one.
        if current_chunk:
            chunks.append(
                current_chunk
            )

        # Add overlap from the previous chunk.
        overlap_text = ""

        if chunks and overlap_characters > 0:
            overlap_text = chunks[-1][
                -overlap_characters:
            ]

        current_chunk = (
            f"{overlap_text} {sentence}".strip()
        )

    if current_chunk:
        chunks.append(
            current_chunk
        )

    return chunks


# =========================================================
# 10. Build section chunks
# =========================================================

def build_section_chunks(
    sections: list[dict[str, Any]],
    document_metadata: dict[str, Any],
    document_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Creates chunks from document sections.

    Every chunk receives the same document-level metadata,
    while section title and page numbers remain chunk-specific.
    """

    chunks: list[dict[str, Any]] = []
    chunk_counter = 1

    for section in sections:
        section_title = section.get(
            "section_title",
            "Unknown Section",
        )

        paragraphs = section.get(
            "paragraphs",
            [],
        )

        if not paragraphs:
            continue

        body_text = " ".join(
            paragraph["content"]
            for paragraph in paragraphs
        )

        full_section_text = normalize_text(
            f"{section_title}. {body_text}"
        )

        if len(full_section_text) < MIN_CHUNK_CHARACTERS:
            continue

        section_parts = split_long_text(
            full_section_text
        )

        page_numbers = sorted(
            {
                paragraph["page_number"]
                for paragraph in paragraphs
                if paragraph.get("page_number") is not None
            }
        )

        for part_number, part_text in enumerate(
            section_parts,
            start=1,
        ):
            # Azure AI Search document keys cannot contain periods.
            # Example:
            # version "3.0" becomes "3_0" only inside chunk_id.
            #
            # We still keep the real version value as "3.0"
            # in the separate metadata field.
            safe_version = str(
                document_metadata["version"]
            ).replace(".", "_")

            chunk_id = (
                f"{document_metadata['policy_id']}"
                f"-v{safe_version}"
                f"-section-{chunk_counter:03d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_type": "section",
                    "section_title": section_title,
                    "section_part": part_number,
                    "content": part_text,
                    "character_count": len(part_text),
                    "page_numbers": page_numbers,
                    "source_file": document_info.get(
                        "source_file"
                    ),
                    "policy_id": document_metadata.get(
                        "policy_id"
                    ),
                    "version": document_metadata.get(
                        "version"
                    ),
                    "status": document_metadata.get(
                        "status"
                    ),
                    "effective_date": document_metadata.get(
                        "effective_date"
                    ),
                    "end_date": document_metadata.get(
                        "end_date"
                    ),
                    "policy_owner": document_metadata.get(
                        "policy_owner"
                    ),
                    "business_unit": document_metadata.get(
                        "business_unit"
                    ),
                    "jurisdiction": document_metadata.get(
                        "jurisdiction"
                    ),
                    "classification": document_metadata.get(
                        "classification"
                    ),
                    "access_scope": document_metadata.get(
                        "access_scope"
                    ),
                    "synthetic_document": document_metadata.get(
                        "synthetic_document"
                    ),
                }
            )

            chunk_counter += 1

    return chunks


# =========================================================
# 11. Rebuild table rows
# =========================================================

def rebuild_table_rows(
    table: dict[str, Any],
) -> list[list[str]]:
    """
    Converts individual table cells into complete rows.
    """

    row_count = table.get(
        "row_count",
        0,
    )

    column_count = table.get(
        "column_count",
        0,
    )

    rows = [
        ["" for _ in range(column_count)]
        for _ in range(row_count)
    ]

    for cell in table.get(
        "cells",
        [],
    ):
        row_index = cell.get(
            "row_index"
        )

        column_index = cell.get(
            "column_index"
        )

        content = normalize_text(
            cell.get(
                "content",
                "",
            )
        )

        if row_index is None or column_index is None:
            continue

        if row_index >= row_count:
            continue

        if column_index >= column_count:
            continue

        rows[row_index][column_index] = content

    return rows


# =========================================================
# 12. Build table chunks
# =========================================================

def build_table_chunks(
    tables: list[dict[str, Any]],
    document_metadata: dict[str, Any],
    document_info: dict[str, Any],
    starting_index: int,
) -> list[dict[str, Any]]:
    """
    Creates one retrieval chunk for each table.

    Tables are kept separate because their row-column meaning
    can be lost when mixed into normal paragraph text.
    """

    chunks: list[dict[str, Any]] = []
    chunk_counter = starting_index

    for table in tables:
        rows = rebuild_table_rows(
            table
        )

        row_texts = []

        for row_number, row in enumerate(
            rows,
            start=1,
        ):
            cleaned_values = [
                value
                for value in row
                if value
            ]

            if not cleaned_values:
                continue

            row_text = (
                f"Row {row_number}: "
                + " | ".join(cleaned_values)
            )

            row_texts.append(
                row_text
            )

        table_text = " ".join(
            row_texts
        )

        if len(table_text) < MIN_CHUNK_CHARACTERS:
            continue

        # Create an Azure AI Search-safe version for the key.
        safe_version = str(
            document_metadata["version"]
        ).replace(".", "_")

        chunk_id = (
            f"{document_metadata['policy_id']}"
            f"-v{safe_version}"
            f"-table-{chunk_counter:03d}"
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": "table",
                "section_title": (
                    f"Table {table.get('table_number')}"
                ),
                "content": table_text,
                "character_count": len(table_text),
                "page_numbers": [],
                "table_number": table.get(
                    "table_number"
                ),
                "row_count": table.get(
                    "row_count"
                ),
                "column_count": table.get(
                    "column_count"
                ),
                "source_file": document_info.get(
                    "source_file"
                ),
                "policy_id": document_metadata.get(
                    "policy_id"
                ),
                "version": document_metadata.get(
                    "version"
                ),
                "status": document_metadata.get(
                    "status"
                ),
                "effective_date": document_metadata.get(
                    "effective_date"
                ),
                "end_date": document_metadata.get(
                    "end_date"
                ),
                "policy_owner": document_metadata.get(
                    "policy_owner"
                ),
                "business_unit": document_metadata.get(
                    "business_unit"
                ),
                "jurisdiction": document_metadata.get(
                    "jurisdiction"
                ),
                "classification": document_metadata.get(
                    "classification"
                ),
                "access_scope": document_metadata.get(
                    "access_scope"
                ),
                "synthetic_document": document_metadata.get(
                    "synthetic_document"
                ),
            }
        )

        chunk_counter += 1

    return chunks


# =========================================================
# 13. Build complete chunk document
# =========================================================

def build_document_chunks(
    metadata_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Creates all section and table chunks for one document.
    """

    document_info = metadata_data.get(
        "document_info",
        {},
    )

    metadata = metadata_data.get(
        "metadata",
        {},
    )

    paragraphs = metadata_data.get(
        "paragraphs",
        [],
    )

    pages = metadata_data.get(
        "pages",
        [],
    )

    tables = metadata_data.get(
        "tables",
        [],
    )

    if not metadata:
        raise ValueError(
            "Document metadata is missing."
        )

    sections = group_paragraphs_by_section(
        paragraphs=paragraphs,
        pages=pages,
    )

    section_chunks = build_section_chunks(
        sections=sections,
        document_metadata=metadata,
        document_info=document_info,
    )

    table_chunks = build_table_chunks(
        tables=tables,
        document_metadata=metadata,
        document_info=document_info,
        starting_index=len(section_chunks) + 1,
    )

    all_chunks = (
        section_chunks
        + table_chunks
    )

    return {
        "document_info": {
            **document_info,
            "processing_stage": "chunked",
        },
        "metadata": metadata,
        "applicability": metadata_data.get(
            "applicability",
            {},
        ),
        "chunk_summary": {
            "section_count": len(sections),
            "section_chunk_count": len(
                section_chunks
            ),
            "table_chunk_count": len(
                table_chunks
            ),
            "total_chunk_count": len(
                all_chunks
            ),
        },
        "chunks": all_chunks,
    }


# =========================================================
# 14. Process one document
# =========================================================

def chunk_document(
    input_path: str | Path,
) -> Path:
    """
    Reads one metadata-enriched JSON file, creates chunks,
    and saves them as a new JSON file.
    """

    input_file = Path(
        input_path
    ).resolve()

    metadata_data = load_json(
        input_file
    )

    chunk_data = build_document_chunks(
        metadata_data
    )

    output_file = (
        CHUNK_OUTPUT_DIR
        / input_file.name
    )

    save_json(
        data=chunk_data,
        output_path=output_file,
    )

    summary = chunk_data[
        "chunk_summary"
    ]

    print(
        f"Sections detected: "
        f"{summary['section_count']}"
    )

    print(
        f"Section chunks created: "
        f"{summary['section_chunk_count']}"
    )

    print(
        f"Table chunks created: "
        f"{summary['table_chunk_count']}"
    )

    print(
        f"Total chunks created: "
        f"{summary['total_chunk_count']}"
    )

    print(
        f"Chunk JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 15. Run directly
# =========================================================

if __name__ == "__main__":
    sample_json = (
        METADATA_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        output_path = chunk_document(
            sample_json
        )

        print(
            "\nSuccess. Document chunks are ready "
            "for PII processing and embeddings."
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
            f"\nChunking failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected chunking error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )