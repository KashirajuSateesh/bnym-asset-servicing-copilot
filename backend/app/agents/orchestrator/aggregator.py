"""
Orchestrator Aggregator

The Orchestrator may receive output from multiple agents:

- Retrieval Agent
- Policy Validation Agent
- Exception Analysis Agent
- Reconciliation Agent

Each agent can return different data.

The Aggregator converts those outputs into one common
evidence package.

Flow:

Agent outputs
    ↓
Normalize
    ↓
Combine evidence
    ↓
Remove duplicates
    ↓
Build one structured evidence package
    ↓
Response Module
"""

from typing import Any


# =========================================================
# 1. Normalize one evidence item
# =========================================================

def normalize_evidence(
    evidence: dict[str, Any],
    source_agent: str,
) -> dict[str, Any]:
    """
    Converts one evidence item into a common structure.

    This makes it easier for the Response Module because
    it receives the same field names regardless of which
    agent produced the evidence.
    """

    return {
        "source_agent": source_agent,

        "chunk_id": evidence.get(
            "chunk_id"
        ),

        "policy_id": evidence.get(
            "policy_id"
        ),

        "version": evidence.get(
            "version"
        ),

        "section_title": evidence.get(
            "section_title"
        ),

        "page_numbers": evidence.get(
            "page_numbers",
            [],
        ),

        "source_file": evidence.get(
            "source_file"
        ),

        "content": evidence.get(
            "content"
        ),

        "search_score": evidence.get(
            "search_score"
        ),

        "rerank_score": evidence.get(
            "rerank_score"
        ),

        "final_rank": evidence.get(
            "final_rank"
        ),
    }


# =========================================================
# 2. Extract evidence from one agent output
# =========================================================

def extract_agent_evidence(
    agent_output: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extracts evidence returned by one specialized agent.

    For now, the Retrieval Agent returns:

    {
        "agent": "retrieval_agent",
        "evidence": [...]
    }

    Later other agents may also return evidence lists.
    """

    if not agent_output:
        return []

    source_agent = agent_output.get(
        "agent",
        "unknown_agent",
    )

    raw_evidence = agent_output.get(
        "evidence",
        [],
    )

    normalized = []

    for item in raw_evidence:

        normalized.append(
            normalize_evidence(
                evidence=item,
                source_agent=source_agent,
            )
        )

    return normalized


# =========================================================
# 3. Remove duplicate evidence
# =========================================================

def deduplicate_evidence(
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Removes duplicate evidence.

    We use chunk_id as the primary duplicate key.

    Later, for MCP/live operational data, we may use:
    - trade_id
    - exception_id
    - break_id
    - source reference
    """

    unique_items = []

    seen_keys = set()

    for item in evidence_items:

        chunk_id = item.get(
            "chunk_id"
        )

        # If chunk_id exists, use it as unique key.
        if chunk_id:

            if chunk_id in seen_keys:
                continue

            seen_keys.add(
                chunk_id
            )

        unique_items.append(
            item
        )

    return unique_items


# =========================================================
# 4. Build final evidence package
# =========================================================

def build_evidence_package(
    query: str,
    routing: dict[str, Any],
    agent_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Combines outputs from one or more specialized agents.

    This becomes the single structured object sent
    from the Orchestrator to the Response Module.
    """

    all_evidence = []

    participating_agents = []

    for agent_output in agent_outputs:

        if not agent_output:
            continue

        agent_name = agent_output.get(
            "agent"
        )

        if agent_name:

            participating_agents.append(
                agent_name
            )

        agent_evidence = (
            extract_agent_evidence(
                agent_output
            )
        )

        all_evidence.extend(
            agent_evidence
        )

    # Remove duplicates before response generation.
    unique_evidence = (
        deduplicate_evidence(
            all_evidence
        )
    )

    return {

        "query": query,

        "routing": routing,

        "participating_agents": (
            list(
                dict.fromkeys(
                    participating_agents
                )
            )
        ),

        "evidence_count": len(
            unique_evidence
        ),

        "evidence": unique_evidence,

        "response_ready": (
            len(unique_evidence) > 0
        ),
    }


# =========================================================
# 5. Local test
# =========================================================

if __name__ == "__main__":

    sample_agent_output = {

        "agent": "retrieval_agent",

        "evidence": [

            {
                "chunk_id":
                    "POL-SEC-001-v3_0-section-001",

                "policy_id":
                    "POL-SEC-001",

                "version":
                    "3.0",

                "section_title":
                    "Purpose and Scope",

                "page_numbers":
                    [1],

                "source_file":
                    "POL-SEC-001.pdf",

                "content":
                    "PII must be masked before AI processing.",

                "search_score":
                    0.04,

                "rerank_score":
                    2.3,

                "final_rank":
                    1,
            }
        ],
    }

    package = build_evidence_package(

        query=(
            "What does the policy say "
            "about masking PII?"
        ),

        routing={

            "intent":
                "document_retrieval",

            "selected_agent":
                "retrieval_agent",

            "confidence":
                0.85,
        },

        agent_outputs=[
            sample_agent_output
        ],
    )

    print(
        "\nEVIDENCE PACKAGE"
    )

    print(
        package
    )