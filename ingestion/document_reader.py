"""
Document Reader

This module reads one local PDF, sends it to Azure AI Document Intelligence,
and saves the extracted content as structured JSON.

Current pipeline stage:

Local PDF
    ↓
Azure AI Document Intelligence
    ↓
Structured raw JSON

Later, this raw JSON will be used by:
- cleaner.py
- chunker.py
- metadata.py
- version_control.py
- pii_processing.py
- embedder.py
- indexer.py
"""

import json
import os
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv


# =========================================================
# 1. Project paths
# =========================================================

# Current file location:
# project_root/ingestion/document_reader.py
#
# Path(__file__) gives the path of this Python file.
# .resolve() converts it into a full absolute path.
# .parent gives ingestion/.
# .parent.parent gives the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# We are currently keeping Azure secrets in backend/.env.
ENV_FILE = PROJECT_ROOT / "backend" / ".env"

# Raw Azure extraction output will be saved here.
RAW_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "raw"


# =========================================================
# 2. Load environment variables
# =========================================================

# Reads backend/.env and loads the values so that Python
# can access them through os.getenv().
load_dotenv(ENV_FILE)

DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv(
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"
)

DOCUMENT_INTELLIGENCE_KEY = os.getenv(
    "AZURE_DOCUMENT_INTELLIGENCE_KEY"
)


# =========================================================
# 3. Validate Azure configuration
# =========================================================

def validate_configuration() -> None:
    """
    Checks whether the required Azure values are available.

    We validate early so that the program gives a clear error
    instead of failing later with a confusing Azure error.
    """

    if not DOCUMENT_INTELLIGENCE_ENDPOINT:
        raise ValueError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is missing "
            "from backend/.env"
        )

    if not DOCUMENT_INTELLIGENCE_KEY:
        raise ValueError(
            "AZURE_DOCUMENT_INTELLIGENCE_KEY is missing "
            "from backend/.env"
        )


# =========================================================
# 4. Create Azure Document Intelligence client
# =========================================================

def create_document_client() -> DocumentIntelligenceClient:
    """
    Creates and returns the Azure Document Intelligence client.

    The client is the Python connection to our Azure
    Document Intelligence resource.
    """

    validate_configuration()

    return DocumentIntelligenceClient(
        endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(
            DOCUMENT_INTELLIGENCE_KEY
        ),
    )


# =========================================================
# 5. Validate the local PDF
# =========================================================

def validate_pdf_path(pdf_path: str | Path) -> Path:
    """
    Validates the local input file.

    Args:
        pdf_path:
            String or Path pointing to the local PDF.

    Returns:
        A validated Path object.

    Raises:
        FileNotFoundError:
            When the file does not exist.

        ValueError:
            When the supplied file is not a PDF.
    """

    file_path = Path(pdf_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file was not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"The supplied path is not a file: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Only PDF files are supported: {file_path}"
        )

    return file_path


# =========================================================
# 6. Helper for paragraph role
# =========================================================

def get_paragraph_role(paragraph: Any) -> str:
    """
    Converts the Azure paragraph role into a simple string.

    Azure may return roles such as:
    - title
    - sectionHeading
    - pageHeader
    - pageFooter
    - pageNumber

    Some paragraphs may not have a role, so we label them
    as 'body'.
    """

    if not paragraph.role:
        return "body"

    # Some Azure SDK values are enum-like objects.
    # The value property usually contains a cleaner string.
    role_value = getattr(
        paragraph.role,
        "value",
        None,
    )

    if role_value:
        return str(role_value)

    return str(paragraph.role)


# =========================================================
# 7. Convert Azure result into project JSON format
# =========================================================

def build_document_json(
    result: Any,
    source_file: Path,
) -> dict:
    """
    Converts the large Azure SDK result into a simpler
    project-specific dictionary.

    We currently preserve:
    - page text
    - page line information
    - paragraph content and roles
    - table cells
    - source file details

    This is called raw JSON because we have not yet removed:
    - repeated headers
    - repeated footers
    - page numbers
    - duplicate table text
    - unnecessary whitespace
    """

    pages: list[dict] = []
    paragraphs: list[dict] = []
    tables: list[dict] = []

    # -----------------------------------------------------
    # 7.1 Process page-level content
    # -----------------------------------------------------

    for page in result.pages or []:
        page_lines: list[dict] = []

        for line_number, line in enumerate(
            page.lines or [],
            start=1,
        ):
            page_lines.append(
                {
                    "line_number": line_number,
                    "content": line.content,
                }
            )

        # Create one complete text block for the page.
        page_text = "\n".join(
            line["content"]
            for line in page_lines
        )

        pages.append(
            {
                "page_number": page.page_number,
                "line_count": len(page_lines),
                "text": page_text,
                "lines": page_lines,
            }
        )

    # -----------------------------------------------------
    # 7.2 Process paragraph-level content
    # -----------------------------------------------------

    for paragraph_number, paragraph in enumerate(
        result.paragraphs or [],
        start=1,
    ):
        paragraphs.append(
            {
                "paragraph_number": paragraph_number,
                "role": get_paragraph_role(
                    paragraph
                ),
                "content": paragraph.content,
            }
        )

    # -----------------------------------------------------
    # 7.3 Process table content
    # -----------------------------------------------------

    for table_number, table in enumerate(
        result.tables or [],
        start=1,
    ):
        cells: list[dict] = []

        # Sorting cells makes the JSON easier to read and
        # helps us later rebuild rows correctly.
        sorted_cells = sorted(
            table.cells or [],
            key=lambda cell: (
                cell.row_index,
                cell.column_index,
            ),
        )

        for cell in sorted_cells:
            cells.append(
                {
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "content": cell.content,
                    "kind": (
                        getattr(cell.kind, "value", None)
                        if cell.kind
                        else None
                    ),
                }
            )

        tables.append(
            {
                "table_number": table_number,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": cells,
            }
        )

    # -----------------------------------------------------
    # 7.4 Build complete document object
    # -----------------------------------------------------

    return {
        "document_info": {
            "source_file": source_file.name,
            "source_path": str(source_file),
            "file_extension": source_file.suffix.lower(),
            "page_count": len(pages),
            "paragraph_count": len(paragraphs),
            "table_count": len(tables),
            "document_intelligence_model": "prebuilt-layout",
            "processing_stage": "raw_extraction",
        },
        "pages": pages,
        "paragraphs": paragraphs,
        "tables": tables,
    }


# =========================================================
# 8. Save JSON to disk
# =========================================================

def save_document_json(
    document_data: dict,
    output_path: Path,
) -> None:
    """
    Saves the extracted document data as formatted JSON.

    Args:
        document_data:
            Dictionary created by build_document_json().

        output_path:
            Destination JSON file path.
    """

    # Create the folder when it does not already exist.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            document_data,
            json_file,
            indent=2,
            ensure_ascii=False,
        )


# =========================================================
# 9. Analyze one local PDF
# =========================================================

def analyze_local_pdf(
    pdf_path: str | Path,
) -> Path:
    """
    Sends one local PDF to Azure Document Intelligence,
    converts the result into structured JSON, and saves it.

    Args:
        pdf_path:
            Path to the local PDF file.

    Returns:
        Path of the generated JSON file.
    """

    # Validate the input PDF.
    file_path = validate_pdf_path(
        pdf_path
    )

    # Create the Azure client.
    document_client = create_document_client()

    print(
        f"Reading document: {file_path.name}"
    )

    # PDFs are binary files, so we open them using "rb".
    with file_path.open("rb") as pdf_file:

        # begin_analyze_document starts an asynchronous Azure job.
        #
        # prebuilt-layout is used because it extracts:
        # - text
        # - page structure
        # - paragraphs
        # - headings
        # - tables
        poller = (
            document_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=pdf_file,
            )
        )

        print(
            "Document submitted to Azure. "
            "Waiting for analysis..."
        )

        # poller.result() waits until Azure finishes.
        result = poller.result()

    # Convert the Azure result into our simplified structure.
    document_data = build_document_json(
        result=result,
        source_file=file_path,
    )

    # Use the same source filename but change .pdf to .json.
    output_file = (
        RAW_OUTPUT_DIR
        / f"{file_path.stem}.json"
    )

    # Save the structured result.
    save_document_json(
        document_data=document_data,
        output_path=output_file,
    )

    # Print only a short summary.
    # We no longer print every line, paragraph, and table.
    print("\nDocument analysis completed.")
    print(
        f"Pages detected: "
        f"{document_data['document_info']['page_count']}"
    )
    print(
        f"Paragraphs detected: "
        f"{document_data['document_info']['paragraph_count']}"
    )
    print(
        f"Tables detected: "
        f"{document_data['document_info']['table_count']}"
    )
    print(
        f"Structured JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 10. Run this file directly
# =========================================================

if __name__ == "__main__":
    """
    This block runs only when we execute:

        uv run python ingestion/document_reader.py

    It does not run when another Python file imports
    analyze_local_pdf().
    """

    # Change this filename when testing a different PDF.
    sample_pdf = (
        PROJECT_ROOT
        / "data"
        / "pdfs"
        / "policies"
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.pdf"
        )
    )

    try:
        generated_json_path = analyze_local_pdf(
            sample_pdf
        )

        print(
            "\nSuccess. Raw extraction is ready "
            "for the cleaning step."
        )
        print(
            f"Output file: {generated_json_path}"
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        # These are expected user/configuration errors.
        print(f"\nValidation error: {error}")

    except Exception as error:
        # This catches unexpected Azure, network,
        # authentication, or SDK errors.
        print(
            "\nDocument processing failed."
        )
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(
            f"Error details: {error}"
        )