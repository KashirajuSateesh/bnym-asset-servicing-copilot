"""
Orchestrator Routing

This module decides which specialized agent should handle
a user query.

Current agents:

- retrieval_agent
- policy_validation_agent
- exception_analysis_agent
- reconciliation_agent

For now, routing is rule-based and simple.

Later, we can upgrade this to:
- LLM-based intent classification
- confidence scoring
- multi-intent routing
- query decomposition
"""

from dataclasses import dataclass


# =========================================================
# 1. Routing result model
# =========================================================

@dataclass
class RoutingDecision:
    """
    Structured result returned by the router.
    """

    agent_name: str
    intent: str
    reason: str
    confidence: float


# =========================================================
# 2. Keyword groups
# =========================================================

RETRIEVAL_KEYWORDS = {
    "policy",
    "sop",
    "procedure",
    "document",
    "requirement",
    "guideline",
    "what does",
    "what are",
    "explain",
}

POLICY_VALIDATION_KEYWORDS = {
    "allowed",
    "permitted",
    "compliant",
    "compliance",
    "violation",
    "valid policy",
    "current policy",
    "exception to policy",
    "can we",
    "should we",
}

EXCEPTION_KEYWORDS = {
    "settlement",
    "settlement exception",
    "failed settlement",
    "trade failed",
    "trade exception",
    "exception",
    "fail",
    "failed",
    "swift",
}

RECONCILIATION_KEYWORDS = {
    "reconciliation",
    "recon",
    "break",
    "cash break",
    "position break",
    "out of balance",
    "difference",
    "mismatch",
}


# =========================================================
# 3. Helper
# =========================================================

def contains_keyword(
    query: str,
    keywords: set[str],
) -> bool:
    """
    Checks whether the query contains any known keyword.
    """

    normalized_query = query.lower()

    return any(
        keyword in normalized_query
        for keyword in keywords
    )


# =========================================================
# 4. Main routing logic
# =========================================================

def route_query(
    query: str,
) -> RoutingDecision:
    """
    Routes one user query to the most appropriate agent.

    Priority matters.

    For example:
    "Is this settlement action allowed under policy?"

    This contains both settlement and policy-validation language.

    We prefer policy_validation_agent because the user is asking
    for a compliance decision, not only operational data.
    """

    if not query or not query.strip():
        raise ValueError(
            "User query cannot be empty."
        )

    normalized_query = query.strip().lower()

    # -----------------------------------------------------
    # 1. Policy validation / compliance
    # -----------------------------------------------------

    if contains_keyword(
        normalized_query,
        POLICY_VALIDATION_KEYWORDS,
    ):
        return RoutingDecision(
            agent_name="policy_validation_agent",
            intent="policy_validation",
            reason=(
                "The query asks whether an action is allowed, "
                "valid, compliant, or permitted under policy."
            ),
            confidence=0.95,
        )

    # -----------------------------------------------------
    # 2. Settlement exception analysis
    # -----------------------------------------------------

    if contains_keyword(
        normalized_query,
        EXCEPTION_KEYWORDS,
    ):
        return RoutingDecision(
            agent_name="exception_analysis_agent",
            intent="settlement_exception_analysis",
            reason=(
                "The query contains settlement, trade failure, "
                "exception, or SWIFT-related language."
            ),
            confidence=0.90,
        )

    # -----------------------------------------------------
    # 3. Reconciliation analysis
    # -----------------------------------------------------

    if contains_keyword(
        normalized_query,
        RECONCILIATION_KEYWORDS,
    ):
        return RoutingDecision(
            agent_name="reconciliation_agent",
            intent="reconciliation_analysis",
            reason=(
                "The query contains reconciliation, break, "
                "balance, mismatch, or position/cash language."
            ),
            confidence=0.90,
        )

    # -----------------------------------------------------
    # 4. Policy / document retrieval
    # -----------------------------------------------------

    if contains_keyword(
        normalized_query,
        RETRIEVAL_KEYWORDS,
    ):
        return RoutingDecision(
            agent_name="retrieval_agent",
            intent="document_retrieval",
            reason=(
                "The query is asking for policy, SOP, procedure, "
                "or document-based information."
            ),
            confidence=0.85,
        )

    # -----------------------------------------------------
    # 5. Default fallback
    # -----------------------------------------------------

    return RoutingDecision(
        agent_name="retrieval_agent",
        intent="general_information",
        reason=(
            "No strong operational intent was detected, "
            "so the query defaults to document retrieval."
        ),
        confidence=0.60,
    )


# =========================================================
# 5. Local test
# =========================================================

if __name__ == "__main__":

    test_queries = [
        "What does the policy say about masking PII?",
        "Is this action allowed under the current policy?",
        "Why did trade TRD-100023 fail settlement?",
        "Why is account ACC-100001 out of balance?",
    ]

    for query in test_queries:

        decision = route_query(
            query
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Agent: {decision.agent_name}"
        )

        print(
            f"Intent: {decision.intent}"
        )

        print(
            f"Confidence: {decision.confidence}"
        )

        print(
            f"Reason: {decision.reason}"
        )