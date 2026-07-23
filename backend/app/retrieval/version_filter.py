"""
Policy Version Filter

This module removes policy chunks that should not be used
for the requested business date.

Rules:
- status must be ACTIVE
- effective_date <= business_date
- end_date >= business_date

Important:
We do NOT simply choose the highest version number.
A future v4 must not beat an active v3 for today's question.
"""

from datetime import date, datetime
from typing import Any


def parse_date(value: str) -> date:
    """
    Converts YYYY-MM-DD text into a Python date.
    """

    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def is_chunk_applicable(
    chunk: dict[str, Any],
    business_date: date,
) -> bool:
    """
    Checks whether one retrieved chunk is valid
    for the supplied business date.
    """

    status = (
        chunk.get("status")
        or ""
    ).upper()

    effective_date = chunk.get(
        "effective_date"
    )

    end_date = chunk.get(
        "end_date"
    )

    if status != "ACTIVE":
        return False

    if not effective_date or not end_date:
        return False

    start = parse_date(
        effective_date
    )

    end = parse_date(
        end_date
    )

    return (
        start
        <= business_date
        <= end
    )


def filter_applicable_versions(
    chunks: list[dict[str, Any]],
    business_date: date | None = None,
) -> list[dict[str, Any]]:
    """
    Keeps only chunks whose policy version is valid
    for the business date.
    """

    business_date = (
        business_date
        or date.today()
    )

    valid_chunks = []

    for chunk in chunks:

        if is_chunk_applicable(
            chunk=chunk,
            business_date=business_date,
        ):
            valid_chunks.append(
                chunk
            )

    return valid_chunks