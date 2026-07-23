"""
Hybrid Retrieval Test

This module tests retrieval from Azure AI Search.

Flow:

User query
    ↓
Generate query embedding with OpenAI
    ↓
Azure AI Search
    ↓
BM25 keyword search
+
Vector similarity search
    ↓
Metadata / policy filters
    ↓
Return Top 15 candidates

Later:
Top 15
    ↓
Reranker
    ↓
Top 5
    ↓
Retrieval Agent
"""

import os
from pathlib import Path
from typing import Any


from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import OpenAI

from datetime import date

from backend.app.retrieval.version_filter import (
    filter_applicable_versions,
)

from backend.app.retrieval.reranker import (
    rerank_results,
)


# =========================================================
# 1. Project paths
# =========================================================

# Current file:
# backend/app/retrieval/search.py
#
# parents[0] -> retrieval
# parents[1] -> app
# parents[2] -> backend
# parents[3] -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = (
    PROJECT_ROOT
    / "backend"
    / ".env"
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

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


# =========================================================
# 3. Retrieval configuration
# =========================================================

# Initial candidate count.
#
# Our architecture is:
#
# Hybrid search -> Top 15 -> Rerank -> Top 5
#
# For now we only implement the first part.
TOP_K_CANDIDATES = 15


# =========================================================
# 4. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Checks that all required service configuration exists.
    """

    required_values = {
        "AZURE_SEARCH_ENDPOINT":
            AZURE_SEARCH_ENDPOINT,

        "AZURE_SEARCH_ADMIN_KEY":
            AZURE_SEARCH_ADMIN_KEY,

        "AZURE_SEARCH_INDEX_NAME":
            AZURE_SEARCH_INDEX_NAME,

        "OPENAI_API_KEY":
            OPENAI_API_KEY,
    }

    missing = [
        name
        for name, value
        in required_values.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing)
        )


# =========================================================
# 5. Create service clients
# =========================================================

def create_search_client() -> SearchClient:
    """
    Creates the Azure AI Search query client.
    """

    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(
            AZURE_SEARCH_ADMIN_KEY
        ),
    )


def create_openai_client() -> OpenAI:
    """
    Creates the OpenAI client.

    We use this only to generate the query embedding.
    """

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# =========================================================
# 6. Generate query embedding
# =========================================================

def generate_query_embedding(
    query: str,
) -> list[float]:
    """
    Converts the user query into an embedding vector.

    This must use the same embedding model used during
    document ingestion.

    Documents:
        text-embedding-3-small

    Query:
        text-embedding-3-small

    Both must exist in the same vector space.
    """

    if not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    client = create_openai_client()

    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=query.strip(),
    )

    return response.data[0].embedding


# =========================================================
# 7. Build optional metadata filter
# =========================================================

def build_metadata_filter(
    policy_id: str | None = None,
    status: str | None = None,
    business_unit: str | None = None,
    jurisdiction: str | None = None,
) -> str | None:
    """
    Builds an Azure AI Search OData filter.

    Example:

    policy_id eq 'POL-SEC-001'
    and status eq 'ACTIVE'

    Later this function will also include:
    - effective-date validation
    - access scope
    - user role / ACL security trimming
    """

    filters: list[str] = []

    if policy_id:
        filters.append(
            f"policy_id eq '{policy_id}'"
        )

    if status:
        filters.append(
            f"status eq '{status}'"
        )

    if business_unit:
        filters.append(
            f"business_unit eq '{business_unit}'"
        )

    if jurisdiction:
        filters.append(
            f"jurisdiction eq '{jurisdiction}'"
        )

    if not filters:
        return None

    return " and ".join(filters)


# =========================================================
# 8. Hybrid search
# =========================================================

def hybrid_search(
    query: str,
    top_k: int = TOP_K_CANDIDATES,
    policy_id: str | None = None,
    status: str | None = None,
    business_unit: str | None = None,
    jurisdiction: str | None = None,
) -> list[dict[str, Any]]:
    """
    Runs hybrid retrieval.

    Hybrid means we send two search signals together:

    1. search_text=query
       -> BM25 / keyword retrieval

    2. vector_queries
       -> semantic vector similarity

    Azure combines the result lists into one ranked result set.
    """

    validate_configuration()

    search_client = create_search_client()

    # -----------------------------------------------------
    # Step 1:
    # Convert user query into the same embedding space
    # as our indexed document chunks.
    # -----------------------------------------------------

    query_embedding = generate_query_embedding(
        query
    )

    print(
        f"Query embedding dimensions: "
        f"{len(query_embedding)}"
    )

    # -----------------------------------------------------
    # Step 2:
    # Configure vector search.
    #
    # k_nearest_neighbors=top_k asks vector search
    # for the strongest semantic candidates.
    #
    # fields="embedding" tells Azure which vector field
    # to compare against.
    # -----------------------------------------------------

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    # -----------------------------------------------------
    # Step 3:
    # Build metadata filtering.
    # -----------------------------------------------------

    metadata_filter = build_metadata_filter(
        policy_id=policy_id,
        status=status,
        business_unit=business_unit,
        jurisdiction=jurisdiction,
    )

    print(
        f"Metadata filter: "
        f"{metadata_filter or 'None'}"
    )

    # -----------------------------------------------------
    # Step 4:
    # Run hybrid search.
    #
    # search_text=query
    #     activates keyword/BM25 search.
    #
    # vector_queries=[vector_query]
    #     activates vector search.
    #
    # Both happen in the same Azure AI Search request.
    # -----------------------------------------------------

    results = search_client.search(

        search_text=query,

        vector_queries=[
            vector_query
        ],

        filter=metadata_filter,

        top=top_k,

        # We return only fields needed by retrieval.
        # We do not return the 1536-value vector.
        select=[
            "chunk_id",
            "safe_content",
            "section_title",
            "policy_id",
            "version",
            "status",
            "effective_date",
            "end_date",
            "business_unit",
            "jurisdiction",
            "classification",
            "access_scope",
            "source_file",
            "page_numbers",
            "chunk_type",
        ],
    )

    # -----------------------------------------------------
    # Step 5:
    # Convert Azure results into normal dictionaries.
    # -----------------------------------------------------

    retrieved_chunks: list[
        dict[str, Any]
    ] = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        retrieved_chunks.append(
            {
                "rank": rank,

                # Azure final hybrid score.
                "search_score": result.get(
                    "@search.score"
                ),

                "chunk_id": result.get(
                    "chunk_id"
                ),

                "content": result.get(
                    "safe_content"
                ),

                "section_title": result.get(
                    "section_title"
                ),

                "policy_id": result.get(
                    "policy_id"
                ),

                "version": result.get(
                    "version"
                ),

                "status": result.get(
                    "status"
                ),

                "effective_date": result.get(
                    "effective_date"
                ),

                "end_date": result.get(
                    "end_date"
                ),

                "business_unit": result.get(
                    "business_unit"
                ),

                "jurisdiction": result.get(
                    "jurisdiction"
                ),

                "classification": result.get(
                    "classification"
                ),

                "access_scope": result.get(
                    "access_scope"
                ),

                "source_file": result.get(
                    "source_file"
                ),

                "page_numbers": result.get(
                    "page_numbers",
                    []
                ),

                "chunk_type": result.get(
                    "chunk_type"
                ),
            }
        )

    return retrieved_chunks


# =========================================================
# 9. Print test results
# =========================================================

def print_results(
    results: list[dict[str, Any]],
) -> None:
    """
    Prints retrieval results in a readable format.

    This is only for development testing.
    """

    if not results:
        print(
            "\nNo retrieval results found."
        )

        return

    print(
        f"\nRetrieved "
        f"{len(results)} candidate chunks."
    )

    for result in results:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FINAL RANK: "
            f"{result.get('final_rank', result.get('rank'))}"
        )

        print(
            f"ORIGINAL HYBRID RANK: "
            f"{result.get('rank')}"
        )

        print(
            f"SCORE: "
            f"{result['search_score']}"
        )

        print(
            f"CHUNK: "
            f"{result['chunk_id']}"
        )

        print(
            f"POLICY: "
            f"{result['policy_id']} "
            f"v{result['version']}"
        )

        print(
            f"STATUS: "
            f"{result['status']}"
        )

        print(
            f"SECTION: "
            f"{result['section_title']}"
        )

        print(
            f"PAGES: "
            f"{result['page_numbers']}"
        )

        # Only show the first 350 characters
        # so terminal output stays readable.
        content = (
            result.get(
                "content"
            )
            or ""
        )

        print(
            "CONTENT:"
        )

        print(
            content[:350]
        )


# =========================================================
# 10. Run directly
# =========================================================

if __name__ == "__main__":

    # This query matches information contained in our
    # Data Access, Privacy, and AI Usage Policy.
    test_query = (
        "What are the requirements for masking PII "
        "before using AI systems and observability traces?"
    )

    try:

        results = hybrid_search(

            query=test_query,

            # For our first test, restrict retrieval
            # to the policy that we indexed.
            policy_id="POL-SEC-001",

            # Only ACTIVE policy versions.
            status="ACTIVE",
        )

        # Step 1:
        # Keep only policy chunks that are valid for today.
        valid_results = filter_applicable_versions(
            chunks=results,
            business_date=date.today(),
        )

        print(
            f"\nValid policy chunks after version filter: "
            f"{len(valid_results)}"
        )

        # Step 2:
        # Rerank the remaining candidates.
        final_results = rerank_results(
            query=test_query,
            chunks=valid_results,
            top_n=5,
        )

        print(
            "\nFINAL TOP 5 RESULTS"
        )

        print_results(
            final_results
        )

    except Exception as error:

        print(
            "\nHybrid retrieval failed."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )