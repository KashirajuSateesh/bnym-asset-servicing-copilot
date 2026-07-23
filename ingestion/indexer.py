"""
Azure AI Search Indexer

This module reads embedding-enriched chunk JSON and uploads
each chunk into Azure AI Search.

Current pipeline:

Embedding JSON
    ↓
Convert chunk to Azure Search document
    ↓
Upload documents in batches
    ↓
Store:
- searchable text
- metadata
- policy version
- access scope
- embedding vector
    ↓
Ready for hybrid search

Later, the Retrieval Agent will query this index using:
- vector search
- BM25 keyword search
- metadata filters
- policy version filters
- semantic ranking
"""

import json
import os
from pathlib import Path
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = (
    PROJECT_ROOT
    / "backend"
    / ".env"
)

EMBEDDING_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
)


# =========================================================
# 2. Load environment variables
# =========================================================

load_dotenv(ENV_FILE)

AZURE_SEARCH_ENDPOINT = os.getenv(
    "AZURE_SEARCH_ENDPOINT"
)

AZURE_SEARCH_ADMIN_KEY = os.getenv(
    "AZURE_SEARCH_ADMIN_KEY"
)

AZURE_SEARCH_INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME",
    "policy-chunks-index",
)


# =========================================================
# 3. Upload configuration
# =========================================================

# We upload documents in small batches during development.
# Later, this can be increased for larger ingestion jobs.
UPLOAD_BATCH_SIZE = 100


# =========================================================
# 4. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Checks that Azure AI Search settings exist.
    """

    if not AZURE_SEARCH_ENDPOINT:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT is missing from backend/.env"
        )

    if not AZURE_SEARCH_ADMIN_KEY:
        raise ValueError(
            "AZURE_SEARCH_ADMIN_KEY is missing from backend/.env"
        )

    if not AZURE_SEARCH_INDEX_NAME:
        raise ValueError(
            "AZURE_SEARCH_INDEX_NAME is missing from backend/.env"
        )


# =========================================================
# 5. Create Azure Search client
# =========================================================

def create_search_client() -> SearchClient:
    """
    Creates a client for uploading and querying documents
    in our Azure AI Search index.
    """

    validate_configuration()

    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(
            AZURE_SEARCH_ADMIN_KEY
        ),
    )


# =========================================================
# 6. Load embedding JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """
    Reads embedding-enriched JSON.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Embedding JSON file was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(
            json_file
        )


# =========================================================
# 7. Build one Azure Search document
# =========================================================

def build_search_document(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """
    Converts one internal chunk into the exact document
    structure expected by Azure AI Search.

    Important:
    We index safe_content for retrieval.

    We do not need to send every internal pipeline field
    into Azure AI Search.
    """

    embedding = chunk.get(
        "embedding"
    )

    if not embedding:
        raise ValueError(
            f"Chunk {chunk.get('chunk_id')} "
            "does not contain an embedding."
        )

    return {
        "chunk_id": chunk.get(
            "chunk_id"
        ),

        # Original document-aware chunk content.
        "content": chunk.get(
            "content",
            "",
        ),

        # PII-safe content used by retrieval and embeddings.
        "safe_content": chunk.get(
            "safe_content",
            "",
        ),

        # Section information.
        "section_title": chunk.get(
            "section_title",
            "",
        ),

        "source_file": chunk.get(
            "source_file",
            "",
        ),

        "page_numbers": chunk.get(
            "page_numbers",
            [],
        ),

        # Policy/version metadata.
        "policy_id": chunk.get(
            "policy_id",
            "",
        ),

        "version": chunk.get(
            "version",
            "",
        ),

        "status": chunk.get(
            "status",
            "",
        ),

        "effective_date": chunk.get(
            "effective_date",
            "",
        ),

        "end_date": chunk.get(
            "end_date",
            "",
        ),

        "policy_owner": chunk.get(
            "policy_owner",
            "",
        ),

        "business_unit": chunk.get(
            "business_unit",
            "",
        ),

        "jurisdiction": chunk.get(
            "jurisdiction",
            "",
        ),

        "classification": chunk.get(
            "classification",
            "",
        ),

        "access_scope": chunk.get(
            "access_scope",
            "",
        ),

        "chunk_type": chunk.get(
            "chunk_type",
            "",
        ),

        # 1536-dimensional vector generated by
        # text-embedding-3-small.
        "embedding": embedding,
    }


# =========================================================
# 8. Prepare all search documents
# =========================================================

def prepare_search_documents(
    embedding_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Converts all embedded chunks into Azure Search documents.
    """

    chunks = embedding_data.get(
        "chunks",
        [],
    )

    if not chunks:
        raise ValueError(
            "No chunks were found in embedding JSON."
        )

    documents = []

    for chunk in chunks:
        documents.append(
            build_search_document(
                chunk
            )
        )

    return documents


# =========================================================
# 9. Create batches
# =========================================================

def create_batches(
    items: list[Any],
    batch_size: int,
) -> list[list[Any]]:
    """
    Splits documents into smaller upload batches.
    """

    return [
        items[index:index + batch_size]
        for index in range(
            0,
            len(items),
            batch_size,
        )
    ]


# =========================================================
# 10. Upload documents
# =========================================================

def upload_documents(
    client: SearchClient,
    documents: list[dict[str, Any]],
) -> None:
    """
    Uploads documents into Azure AI Search.

    Azure returns one result per document.
    We check each result so failed uploads are visible.
    """

    batches = create_batches(
        items=documents,
        batch_size=UPLOAD_BATCH_SIZE,
    )

    total_success = 0
    total_failed = 0

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"Uploading batch "
            f"{batch_number} of {len(batches)}..."
        )

        results = client.upload_documents(
            documents=batch
        )

        for result in results:
            if result.succeeded:
                total_success += 1
            else:
                total_failed += 1

                print(
                    "Upload failed for document: "
                    f"{result.key}"
                )

                print(
                    f"Error: {result.error_message}"
                )

    print(
        f"\nDocuments uploaded successfully: "
        f"{total_success}"
    )

    print(
        f"Documents failed: "
        f"{total_failed}"
    )

    if total_failed > 0:
        raise RuntimeError(
            f"{total_failed} documents failed to index."
        )


# =========================================================
# 11. Verify indexed document count
# =========================================================

def verify_index_count(
    client: SearchClient,
) -> int:
    """
    Gets the current document count from Azure AI Search.

    This confirms that documents exist in the index.
    """

    count = client.get_document_count()

    print(
        f"Current documents in index: {count}"
    )

    return count


# =========================================================
# 12. Index one embedding document
# =========================================================

def index_document(
    input_path: str | Path,
) -> None:
    """
    Reads one embedding JSON file and uploads all chunks
    into Azure AI Search.
    """

    input_file = Path(
        input_path
    ).resolve()

    embedding_data = load_json(
        input_file
    )

    documents = prepare_search_documents(
        embedding_data
    )

    client = create_search_client()

    print(
        f"Index name: {AZURE_SEARCH_INDEX_NAME}"
    )

    print(
        f"Chunks ready for indexing: "
        f"{len(documents)}"
    )

    upload_documents(
        client=client,
        documents=documents,
    )

    verify_index_count(
        client
    )


# =========================================================
# 13. Run directly
# =========================================================

if __name__ == "__main__":

    sample_json = (
        EMBEDDING_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        index_document(
            sample_json
        )

        print(
            "\nSuccess. Azure AI Search indexing completed."
        )

        print(
            "The index is now ready for retrieval testing."
        )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            f"\nIndexing failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected indexing error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )