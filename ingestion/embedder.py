"""
Embedding Generation Module

This module reads PII-safe chunks, sends each chunk's safe content
to the OpenAI Embeddings API, and saves the returned vectors.

Current pipeline:

PII-safe chunk JSON
        ↓
Select safe_content
        ↓
OpenAI text-embedding-3-small
        ↓
Embedding vector
        ↓
Save embedding-enriched JSON

The generated JSON will later be indexed into Azure AI Search.

Important:
- We send only safe_content to the embedding API.
- We do not send the original unmasked content.
- We process chunks in batches to reduce API calls.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = (
    PROJECT_ROOT
    / "backend"
    / ".env"
)

PII_SAFE_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pii_safe"
)

EMBEDDING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
)


# =========================================================
# 2. Load environment variables
# =========================================================

load_dotenv(ENV_FILE)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


# =========================================================
# 3. Embedding configuration
# =========================================================

# Number of chunks sent in one request.
#
# A small batch is easier to debug during development.
# We can increase this later.
EMBEDDING_BATCH_SIZE = 20

# Number of times to retry when the API call fails.
MAX_RETRIES = 3

# Wait time before retrying.
RETRY_DELAY_SECONDS = 2


# =========================================================
# 4. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Confirms that the required OpenAI API key exists.

    We stop early with a clear message when the key
    is missing from backend/.env.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )


# =========================================================
# 5. Create OpenAI client
# =========================================================

def create_openai_client() -> OpenAI:
    """
    Creates the OpenAI API client.

    The client is used to call the embeddings endpoint.
    """

    validate_configuration()

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# =========================================================
# 6. Load JSON
# =========================================================

def load_json(
    input_path: Path,
) -> dict[str, Any]:
    """
    Reads the PII-safe chunk JSON file.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"PII-safe JSON file was not found: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as json_file:
        return json.load(
            json_file
        )


# =========================================================
# 7. Save JSON
# =========================================================

def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Saves embedding-enriched JSON.
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
# 8. Split items into batches
# =========================================================

def create_batches(
    items: list[Any],
    batch_size: int,
) -> list[list[Any]]:
    """
    Divides a list into smaller batches.

    Example:

        45 chunks with batch size 20

    Becomes:

        batch 1 -> 20 chunks
        batch 2 -> 20 chunks
        batch 3 -> 5 chunks
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
# 9. Validate chunks before embedding
# =========================================================

def prepare_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Validates chunks and keeps only chunks that are safe
    and useful for embedding.

    A chunk is included only when:
    - safe_for_embedding is True
    - safe_content is not empty
    """

    prepared_chunks = []

    for chunk in chunks:
        pii_processing = chunk.get(
            "pii_processing",
            {},
        )

        safe_for_embedding = pii_processing.get(
            "safe_for_embedding",
            False,
        )

        safe_content = chunk.get(
            "safe_content",
            "",
        ).strip()

        if not safe_for_embedding:
            print(
                "Skipping unsafe chunk: "
                f"{chunk.get('chunk_id')}"
            )
            continue

        if not safe_content:
            print(
                "Skipping empty chunk: "
                f"{chunk.get('chunk_id')}"
            )
            continue

        prepared_chunks.append(
            chunk
        )

    return prepared_chunks


# =========================================================
# 10. Generate embeddings for one batch
# =========================================================

def generate_batch_embeddings(
    client: OpenAI,
    texts: list[str],
) -> list[list[float]]:
    """
    Sends one batch of text to the OpenAI Embeddings API.

    Args:
        client:
            OpenAI API client.

        texts:
            List of safe chunk strings.

    Returns:
        List of embedding vectors.

    The returned vector order matches the input text order.
    """

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts,
            )

            # Sort by index to ensure vectors remain
            # aligned with input texts.
            sorted_results = sorted(
                response.data,
                key=lambda item: item.index,
            )

            return [
                item.embedding
                for item in sorted_results
            ]

        except Exception as error:
            last_error = error

            print(
                f"Embedding attempt {attempt} failed: "
                f"{error}"
            )

            if attempt < MAX_RETRIES:
                wait_time = (
                    RETRY_DELAY_SECONDS
                    * attempt
                )

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

    raise RuntimeError(
        "Embedding request failed after "
        f"{MAX_RETRIES} attempts."
    ) from last_error


# =========================================================
# 11. Add embeddings to chunks
# =========================================================

def embed_chunks(
    chunks: list[dict[str, Any]],
    client: OpenAI,
) -> list[dict[str, Any]]:
    """
    Generates embeddings for all valid chunks.

    Each chunk receives:
    - embedding vector
    - embedding model
    - vector dimensions
    """

    prepared_chunks = prepare_chunks(
        chunks
    )

    if not prepared_chunks:
        raise ValueError(
            "No safe chunks were available for embedding."
        )

    batches = create_batches(
        items=prepared_chunks,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    embedded_chunks = []

    for batch_number, batch in enumerate(
        batches,
        start=1,
    ):
        print(
            f"Processing embedding batch "
            f"{batch_number} of {len(batches)}..."
        )

        texts = [
            chunk["safe_content"]
            for chunk in batch
        ]

        vectors = generate_batch_embeddings(
            client=client,
            texts=texts,
        )

        if len(vectors) != len(batch):
            raise ValueError(
                "Embedding response count does not match "
                "the number of input chunks."
            )

        for chunk, vector in zip(
            batch,
            vectors,
        ):
            enriched_chunk = chunk.copy()

            enriched_chunk["embedding"] = (
                vector
            )

            enriched_chunk[
                "embedding_model"
            ] = OPENAI_EMBEDDING_MODEL

            enriched_chunk[
                "embedding_dimensions"
            ] = len(vector)

            embedded_chunks.append(
                enriched_chunk
            )

    return embedded_chunks


# =========================================================
# 12. Build embedding-enriched document
# =========================================================

def build_embedding_document(
    pii_safe_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Creates the final embedding-enriched document structure.
    """

    chunks = pii_safe_data.get(
        "chunks",
        [],
    )

    if not chunks:
        raise ValueError(
            "No chunks were found in the PII-safe document."
        )

    client = create_openai_client()

    embedded_chunks = embed_chunks(
        chunks=chunks,
        client=client,
    )

    document_info = pii_safe_data.get(
        "document_info",
        {},
    ).copy()

    document_info["processing_stage"] = (
        "embedded"
    )

    return {
        "document_info": document_info,
        "metadata": pii_safe_data.get(
            "metadata",
            {},
        ),
        "applicability": pii_safe_data.get(
            "applicability",
            {},
        ),
        "chunk_summary": pii_safe_data.get(
            "chunk_summary",
            {},
        ),
        "pii_summary": pii_safe_data.get(
            "pii_summary",
            {},
        ),
        "embedding_summary": {
            "embedding_model": (
                OPENAI_EMBEDDING_MODEL
            ),
            "total_embedded_chunks": len(
                embedded_chunks
            ),
            "embedding_dimensions": (
                embedded_chunks[0][
                    "embedding_dimensions"
                ]
                if embedded_chunks
                else 0
            ),
            "embedding_source_field": (
                "safe_content"
            ),
        },
        "chunks": embedded_chunks,
    }


# =========================================================
# 13. Process one document
# =========================================================

def generate_document_embeddings(
    input_path: str | Path,
) -> Path:
    """
    Reads one PII-safe JSON file, generates embeddings,
    and saves the enriched result.
    """

    input_file = Path(
        input_path
    ).resolve()

    pii_safe_data = load_json(
        input_file
    )

    embedding_data = build_embedding_document(
        pii_safe_data
    )

    output_file = (
        EMBEDDING_OUTPUT_DIR
        / input_file.name
    )

    save_json(
        data=embedding_data,
        output_path=output_file,
    )

    summary = embedding_data[
        "embedding_summary"
    ]

    print(
        f"\nEmbedding model: "
        f"{summary['embedding_model']}"
    )

    print(
        f"Chunks embedded: "
        f"{summary['total_embedded_chunks']}"
    )

    print(
        f"Embedding dimensions: "
        f"{summary['embedding_dimensions']}"
    )

    print(
        f"Embedding JSON saved to: {output_file}"
    )

    return output_file


# =========================================================
# 14. Run directly
# =========================================================

if __name__ == "__main__":
    sample_json = (
        PII_SAFE_INPUT_DIR
        / (
            "POL-SEC-001_data_access,_privacy,"
            "_and_ai_usage_policy_v3.0.json"
        )
    )

    try:
        output_path = generate_document_embeddings(
            sample_json
        )

        print(
            "\nSuccess. Embedded chunks are ready "
            "for Azure AI Search indexing."
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
            f"\nEmbedding generation failed: {error}"
        )

    except Exception as error:
        print(
            "\nUnexpected embedding error."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )