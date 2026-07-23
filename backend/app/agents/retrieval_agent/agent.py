"""
Retrieval Agent

This agent is responsible for finding trusted policy context
from Azure AI Search.

Flow:

User query
    ↓
Hybrid retrieval
    ↓
Policy/version validation
    ↓
Reranking
    ↓
Top 5 trusted chunks
    ↓
Return structured evidence to Orchestrator

Important:
The Retrieval Agent does NOT generate the final answer.
It only returns evidence.
"""

from datetime import date
from typing import Any

from backend.app.retrieval.search import hybrid_search
from backend.app.retrieval.version_filter import (
    filter_applicable_versions,
)
from backend.app.retrieval.reranker import (
    rerank_results,
)


FINAL_RESULT_COUNT = 5


class RetrievalAgent:
    """
    Retrieves policy evidence for the Orchestrator.
    """

    def retrieve(
        self,
        query: str,
        policy_id: str | None = None,
        business_date: date | None = None,
    ) -> dict[str, Any]:
        """
        Runs the complete retrieval pipeline.

        Returns structured evidence instead of a final
        natural-language answer.
        """

        # Step 1:
        # Hybrid search gives us broad candidate evidence.
        candidates = hybrid_search(
            query=query,
            top_k=15,
            policy_id=policy_id,
        )

        # Step 2:
        # Keep only policy versions valid for the requested date.
        valid_candidates = filter_applicable_versions(
            chunks=candidates,
            business_date=business_date or date.today(),
        )

        # Step 3:
        # Rerank and keep the strongest Top 5 chunks.
        final_chunks = rerank_results(
            query=query,
            chunks=valid_candidates,
            top_n=FINAL_RESULT_COUNT,
        )

        # Step 4:
        # Return a structured package.
        #
        # Later the Orchestrator will combine this with outputs
        # from Policy, Exception, and Reconciliation agents.
        return {
            "agent": "retrieval_agent",
            "query": query,
            "candidate_count": len(candidates),
            "valid_candidate_count": len(valid_candidates),
            "evidence_count": len(final_chunks),
            "evidence": final_chunks,
        }


# =========================================================
# Local test
# =========================================================

if __name__ == "__main__":

    agent = RetrievalAgent()

    test_query = (
        "What are the requirements for masking PII "
        "before sending data to AI systems?"
    )

    result = agent.retrieve(
        query=test_query,
        policy_id="POL-SEC-001",
    )

    print(
        f"\nAgent: {result['agent']}"
    )

    print(
        f"Candidates: {result['candidate_count']}"
    )

    print(
        f"Valid candidates: "
        f"{result['valid_candidate_count']}"
    )

    print(
        f"Final evidence: "
        f"{result['evidence_count']}"
    )

    for item in result["evidence"]:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Rank: "
            f"{item.get('final_rank')}"
        )

        print(
            f"Policy: "
            f"{item.get('policy_id')} "
            f"v{item.get('version')}"
        )

        print(
            f"Section: "
            f"{item.get('section_title')}"
        )

        print(
            f"Pages: "
            f"{item.get('page_numbers')}"
        )

        content = (
            item.get("content")
            or ""
        )

        print(
            content[:400]
        )