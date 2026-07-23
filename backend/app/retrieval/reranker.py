"""
Simple Retrieval Reranker

This module reranks the filtered Top 15 chunks
and returns the strongest Top 5.

Current scoring considers:
- Azure hybrid score
- query overlap with section title
- query overlap with chunk content
"""

import re
from typing import Any


TOP_FINAL_RESULTS = 5


def tokenize(
    text: str,
) -> set[str]:
    """
    Converts text into lowercase word tokens.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower(),
        )
    )


def calculate_overlap_score(
    query: str,
    text: str,
) -> float:
    """
    Calculates simple keyword overlap.
    """

    query_tokens = tokenize(
        query
    )

    text_tokens = tokenize(
        text
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & text_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def rerank_results(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = TOP_FINAL_RESULTS,
) -> list[dict[str, Any]]:
    """
    Reranks retrieved chunks and returns Top N.
    """

    reranked = []

    for chunk in chunks:

        azure_score = float(
            chunk.get(
                "search_score"
            )
            or 0.0
        )

        section_score = (
            calculate_overlap_score(
                query=query,
                text=chunk.get(
                    "section_title",
                    "",
                ),
            )
        )

        content_score = (
            calculate_overlap_score(
                query=query,
                text=chunk.get(
                    "content",
                    "",
                ),
            )
        )

        # Weighted reranking score.
        final_score = (
            azure_score
            + (section_score * 2.0)
            + (content_score * 1.5)
        )

        enriched = (
            chunk.copy()
        )

        enriched[
            "rerank_score"
        ] = final_score

        reranked.append(
            enriched
        )

    reranked.sort(
        key=lambda item: (
            item["rerank_score"]
        ),
        reverse=True,
    )

    final_results = (
        reranked[:top_n]
    )

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        result[
            "final_rank"
        ] = rank

    return final_results