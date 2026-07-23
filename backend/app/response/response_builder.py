"""
Grounded Response Builder

This module receives ONE aggregated evidence package
from the Orchestrator.

Flow:

Orchestrator Aggregator
        ↓
Structured Evidence Package
        ↓
Build grounded prompt
        ↓
OpenAI response generation
        ↓
Return answer + citations

Important:
The Response Module must answer only from supplied evidence.
It should not invent policy details.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from backend.app.response.citations import (
    attach_citations,
)



# =========================================================
# 1. Configuration
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

ENV_FILE = (
    PROJECT_ROOT
    / "backend"
    / ".env"
)

load_dotenv(
    ENV_FILE
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

OPENAI_RESPONSE_MODEL = os.getenv(
    "OPENAI_RESPONSE_MODEL",
    "gpt-4o-mini",
)


# =========================================================
# 2. Validate configuration
# =========================================================

def validate_configuration() -> None:
    """
    Confirms that the OpenAI key exists.
    """

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing from backend/.env"
        )


# =========================================================
# 3. Create OpenAI client
# =========================================================

def create_openai_client() -> OpenAI:
    """
    Creates the OpenAI client.
    """

    validate_configuration()

    return OpenAI(
        api_key=OPENAI_API_KEY
    )


# =========================================================
# 4. Build evidence text
# =========================================================

def build_evidence_text(
    evidence_items: list[dict[str, Any]],
) -> str:
    """
    Converts structured evidence into prompt context.
    """

    sections = []

    for index, item in enumerate(
        evidence_items,
        start=1,
    ):

        sections.append(
            (
                f"EVIDENCE {index}\n"
                f"Citation: {item.get('citation')}\n"
                f"Content: {item.get('content', '')}\n"
            )
        )

    return "\n".join(
        sections
    )


# =========================================================
# 5. Build grounded prompt
# =========================================================

def build_prompt(
    query: str,
    evidence_items: list[dict[str, Any]],
) -> str:
    """
    Builds a strict grounded-generation prompt.
    """

    evidence_text = build_evidence_text(
        evidence_items
    )

    return f"""
You are an Asset Servicing AI Copilot.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not invent facts.
2. Do not use unsupported outside knowledge.
3. If evidence is insufficient, clearly say so.
4. Use simple, clear business language.
5. Include citations directly after the statements they support.
6. Use only the citation labels provided in the evidence.
7. Do not create fake citations.
8. Prefer the most directly relevant evidence.

USER QUESTION:
{query}

EVIDENCE:
{evidence_text}

FINAL ANSWER:
""".strip()


# =========================================================
# 6. Generate grounded answer
# =========================================================

def generate_grounded_answer(
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Generates the final grounded response.

    Input:
        Aggregated evidence package from Orchestrator.

    Output:
        Final answer + citations + evidence metadata.
    """

    query = evidence_package.get(
        "query",
        "",
    )

    raw_evidence = evidence_package.get(
        "evidence",
        [],
    )

    if not query:
        raise ValueError(
            "Evidence package does not contain a query."
        )

    if not raw_evidence:
        return {
            "answer": (
                "I could not find enough trusted evidence "
                "to answer this question."
            ),
            "citations": [],
            "evidence_count": 0,
            "grounded": False,
        }

    evidence_with_citations = (
        attach_citations(
            raw_evidence
        )
    )

    prompt = build_prompt(
        query=query,
        evidence_items=evidence_with_citations,
    )

    client = create_openai_client()

    response = client.chat.completions.create(

        model=OPENAI_RESPONSE_MODEL,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded enterprise AI assistant. "
                    "Use only supplied evidence."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = (
        response.choices[0]
        .message
        .content
        or ""
    ).strip()

    return {
        "answer": answer,

        "citations": [
            item["citation"]
            for item in evidence_with_citations
        ],

        "evidence_count": len(
            evidence_with_citations
        ),

        "grounded": True,

        "participating_agents": (
            evidence_package.get(
                "participating_agents",
                [],
            )
        ),
    }


# =========================================================
# 7. Local test
# =========================================================

if __name__ == "__main__":

    sample_package = {

        "query": (
            "What does the policy require "
            "for PII before using AI systems?"
        ),

        "participating_agents": [
            "retrieval_agent"
        ],

        "evidence": [

            {
                "policy_id":
                    "POL-SEC-001",

                "version":
                    "3.0",

                "section_title":
                    "Current Version-Specific Rules",

                "page_numbers":
                    [1],

                "content":
                    (
                        "PII masking is required before "
                        "non-essential LLM prompts and "
                        "observability traces."
                    ),
            },

            {
                "policy_id":
                    "POL-SEC-001",

                "version":
                    "3.0",

                "section_title":
                    "Audit, Privacy, and AI Requirements",

                "page_numbers":
                    [2],

                "content":
                    (
                        "Mask PII and confidential fields "
                        "in prompts and traces unless the "
                        "value is explicitly required and permitted."
                    ),
            },
        ],
    }

    try:

        result = generate_grounded_answer(
            sample_package
        )

        print(
            "\nFINAL ANSWER\n"
        )

        print(
            result["answer"]
        )

        print(
            "\nCITATIONS\n"
        )

        for citation in result[
            "citations"
        ]:

            print(
                citation
            )

    except Exception as error:

        print(
            "\nResponse generation failed."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: "
            f"{error}"
        )