"""
Create Azure AI Search Index

This script creates the search index used by the Retrieval Agent.

The index supports:

- BM25 keyword search
- Vector search
- Metadata filtering
- Policy version filtering
- Semantic ranking

The embedding dimension is 1536 because we are using:
text-embedding-3-small
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


# =========================================================
# 3. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Make sure all Azure AI Search settings exist
    before trying to create the index.
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
# 4. Create index client
# =========================================================

def create_index_client() -> SearchIndexClient:
    """
    Creates the Azure AI Search index-management client.

    This client is used to:
    - create indexes
    - update indexes
    - delete indexes
    """

    validate_configuration()

    return SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(
            AZURE_SEARCH_ADMIN_KEY
        ),
    )


# =========================================================
# 5. Define index fields
# =========================================================

def build_index_fields() -> list:
    """
    Defines all fields stored in Azure AI Search.

    Some fields are searchable.
    Some are filterable.
    Some are used for vector similarity.
    """

    return [

        # -------------------------------------------------
        # Unique chunk identifier
        # -------------------------------------------------

        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),

        # -------------------------------------------------
        # Main searchable content
        # -------------------------------------------------

        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),

        SearchableField(
            name="safe_content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),

        # -------------------------------------------------
        # Section / source information
        # -------------------------------------------------

        SearchableField(
            name="section_title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),

        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="page_numbers",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Int32
            ),
            filterable=True,
        ),

        # -------------------------------------------------
        # Policy metadata
        # -------------------------------------------------

        SimpleField(
            name="policy_id",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),

        SimpleField(
            name="version",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),

        SimpleField(
            name="status",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="effective_date",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),

        SimpleField(
            name="end_date",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),

        SimpleField(
            name="policy_owner",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="business_unit",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="jurisdiction",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="classification",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        SimpleField(
            name="access_scope",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        # -------------------------------------------------
        # Chunk type
        # -------------------------------------------------

        SimpleField(
            name="chunk_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),

        # -------------------------------------------------
        # Vector embedding
        # -------------------------------------------------

        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,

            # text-embedding-3-small produces
            # 1536 dimensions by default.
            vector_search_dimensions=1536,

            vector_search_profile_name=(
                "policy-vector-profile"
            ),
        ),
    ]


# =========================================================
# 6. Configure vector search
# =========================================================

def build_vector_search() -> VectorSearch:
    """
    Configures HNSW vector search.

    HNSW is an approximate nearest-neighbor algorithm.

    It allows Azure AI Search to efficiently find chunks
    whose embedding vectors are semantically similar
    to the user's query vector.
    """

    return VectorSearch(

        algorithms=[
            HnswAlgorithmConfiguration(
                name="policy-hnsw",
            )
        ],

        profiles=[
            VectorSearchProfile(
                name="policy-vector-profile",
                algorithm_configuration_name=(
                    "policy-hnsw"
                ),
            )
        ],
    )


# =========================================================
# 7. Configure semantic ranking
# =========================================================

def build_semantic_search() -> SemanticSearch:
    """
    Defines which text fields semantic ranking should use.

    Semantic ranking will later help rerank the initial
    keyword/vector results.
    """

    semantic_configuration = (
        SemanticConfiguration(

            name="policy-semantic-config",

            prioritized_fields=(
                SemanticPrioritizedFields(

                    title_field=SemanticField(
                        field_name="section_title"
                    ),

                    content_fields=[
                        SemanticField(
                            field_name="safe_content"
                        )
                    ],
                )
            ),
        )
    )

    return SemanticSearch(
        configurations=[
            semantic_configuration
        ]
    )


# =========================================================
# 8. Build complete index definition
# =========================================================

def build_search_index() -> SearchIndex:
    """
    Combines:

    - index fields
    - vector search configuration
    - semantic search configuration
    """

    return SearchIndex(

        name=AZURE_SEARCH_INDEX_NAME,

        fields=build_index_fields(),

        vector_search=build_vector_search(),

        semantic_search=build_semantic_search(),
    )


# =========================================================
# 9. Create or update the index
# =========================================================

def create_or_update_index() -> None:
    """
    Creates the index in Azure AI Search.

    create_or_update_index is useful because we can run
    this script again later if the schema changes.
    """

    client = create_index_client()

    index = build_search_index()

    print(
        f"Creating or updating index: "
        f"{AZURE_SEARCH_INDEX_NAME}"
    )

    result = client.create_or_update_index(
        index
    )

    print(
        "\nAzure AI Search index created successfully."
    )

    print(
        f"Index name: {result.name}"
    )


# =========================================================
# 10. Run directly
# =========================================================

if __name__ == "__main__":

    try:

        create_or_update_index()

    except Exception as error:

        print(
            "\nSearch index creation failed."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: {error}"
        )