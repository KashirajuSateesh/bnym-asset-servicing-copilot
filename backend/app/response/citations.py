"""
Citation Builder

This module converts retrieved evidence into simple,
consistent citation labels for the final answer.

Example:
[POL-SEC-001 v3.0, page 2, Audit, Privacy, and AI Requirements]
"""

from typing import Any


def build_citation(
    evidence: dict[str, Any],
) -> str:
    """
    Builds one readable citation from evidence metadata.
    """

    policy_id = evidence.get(
        "policy_id",
        "UNKNOWN_POLICY",
    )

    version = evidence.get(
        "version",
        "UNKNOWN_VERSION",
    )

    section_title = evidence.get(
        "section_title",
        "Unknown Section",
    )

    page_numbers = evidence.get(
        "page_numbers",
        [],
    )

    if page_numbers:
        page_text = ", ".join(
            str(page)
            for page in page_numbers
        )

        page_label = (
            f"page {page_text}"
            if len(page_numbers) == 1
            else f"pages {page_text}"
        )

    else:
        page_label = "page unknown"

    return (
        f"[{policy_id} v{version}, "
        f"{page_label}, "
        f"{section_title}]"
    )


def attach_citations(
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Adds a citation string to each evidence item.
    """

    enriched = []

    for item in evidence_items:

        copied_item = item.copy()

        copied_item["citation"] = (
            build_citation(
                item
            )
        )

        enriched.append(
            copied_item
        )

    return enriched