"""
Main Orchestrator Agent

The Orchestrator is the manager of the multi-agent system.

Responsibilities:

1. Receive the user query
2. Classify the intent
3. Decide which specialized agent should handle it
4. Call that agent
5. Collect the returned evidence
6. Later aggregate outputs from multiple agents
7. Send one structured package to the Response Module

Important:

Specialized agents do NOT call the Response Module directly.

Flow:

User
  ↓
Orchestrator
  ↓
Router
  ↓
Specialized Agent
  ↓
Structured Evidence
  ↓
Orchestrator
"""

from typing import Any

from backend.app.agents.orchestrator.routing import (
    RoutingDecision,
    route_query,
)

from backend.app.agents.retrieval_agent.agent import (
    RetrievalAgent,
)

from backend.app.agents.orchestrator.aggregator import (
    build_evidence_package,
)
from backend.app.response.response_builder import (
    generate_grounded_answer,
)


class OrchestratorAgent:
    """
    Main manager agent.

    For now, only the Retrieval Agent is fully connected.

    Other agents will be connected one by one:
    - Policy Validation Agent
    - Exception Analysis Agent
    - Reconciliation Agent
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize available specialized agents.
        """

        self.retrieval_agent = (
            RetrievalAgent()
        )


    # =====================================================
    # 1. Execute Retrieval Agent
    # =====================================================

    def _run_retrieval_agent(
        self,
        query: str,
        policy_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Calls the Retrieval Agent.

        The Retrieval Agent performs:

        Query
            ↓
        Query embedding
            ↓
        Hybrid Azure AI Search
            ↓
        Version validation
            ↓
        Reranking
            ↓
        Top 5 evidence chunks
        """

        return self.retrieval_agent.retrieve(
            query=query,
            policy_id=policy_id,
        )


    # =====================================================
    # 2. Handle unsupported agents temporarily
    # =====================================================

    def _agent_not_connected(
        self,
        decision: RoutingDecision,
    ) -> dict[str, Any]:
        """
        Temporary response for agents that have not yet
        been implemented.

        This lets us test routing without pretending that
        the specialized agent already exists.
        """

        return {

            "status": "agent_not_connected",

            "agent": decision.agent_name,

            "intent": decision.intent,

            "message": (
                f"{decision.agent_name} was selected by "
                "the router but has not been connected yet."
            ),
        }


    # =====================================================
    # 3. Main Orchestrator execution
    # =====================================================

    def run(
        self,
        query: str,
        policy_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point for the Orchestrator.

        Flow:

        User Query
            ↓
        Routing Decision
            ↓
        Specialized Agent
            ↓
        Agent Evidence
            ↓
        Orchestrator Result
        """

        if not query or not query.strip():

            raise ValueError(
                "User query cannot be empty."
            )

        # -------------------------------------------------
        # Step 1:
        # Determine the user intent.
        # -------------------------------------------------

        decision = route_query(
            query
        )

        print(
            "\nORCHESTRATOR ROUTING"
        )

        print(
            f"Intent: "
            f"{decision.intent}"
        )

        print(
            f"Selected agent: "
            f"{decision.agent_name}"
        )

        print(
            f"Routing confidence: "
            f"{decision.confidence}"
        )


        # -------------------------------------------------
        # Step 2:
        # Call the selected specialized agent.
        # -------------------------------------------------

        if (
            decision.agent_name
            == "retrieval_agent"
        ):

            agent_output = (
                self._run_retrieval_agent(
                    query=query,
                    policy_id=policy_id,
                )
            )

        elif decision.agent_name in {

            "policy_validation_agent",

            "exception_analysis_agent",

            "reconciliation_agent",

        }:

            agent_output = (
                self._agent_not_connected(
                    decision
                )
            )

        else:

            raise ValueError(
                "Unknown agent selected: "
                f"{decision.agent_name}"
            )


        # -------------------------------------------------
        # Step 3:
        # Aggregate all specialized-agent outputs into one
        # common evidence package.
        # -------------------------------------------------

        routing_info = {
            "intent": decision.intent,
            "selected_agent": decision.agent_name,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }

        evidence_package = build_evidence_package(
            query=query,
            routing=routing_info,
            agent_outputs=[
                agent_output
            ],
        )

        # -------------------------------------------------
        # Step 4:
        # Send the ONE aggregated evidence package
        # to the Response Module.
        # -------------------------------------------------

        final_response = generate_grounded_answer(
            evidence_package
        )

        # -------------------------------------------------
        # Step 5:
        # Return the orchestrator result.
        #
        # Later the Response Module will consume:
        # evidence_package
        # -------------------------------------------------

        return {

            "orchestrator": {
                "query": query,
                "routing": routing_info,
            },

            "evidence_package": (
                evidence_package
            ),

            "response": (
                final_response
            ),
        }


# =========================================================
# Local Test
# =========================================================

if __name__ == "__main__":

    orchestrator = (
        OrchestratorAgent()
    )

    test_query = (
        "What does the policy say about "
        "masking PII before using AI systems?"
    )

    try:

        result = orchestrator.run(

            query=test_query,

            policy_id=(
                "POL-SEC-001"
            ),
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ORCHESTRATOR RESULT"
        )

        print(
            "=" * 70
        )

        routing = (
            result[
                "orchestrator"
            ][
                "routing"
            ]
        )

        print(
            f"\nSelected Agent: "
            f"{routing['selected_agent']}"
        )

        print(
            f"Intent: "
            f"{routing['intent']}"
        )

        evidence_package = (
            result[
                "evidence_package"
            ]
        )

        print(
            f"\nParticipating Agents: "
            f"{evidence_package['participating_agents']}"
        )

        print(
            f"Evidence Count: "
            f"{evidence_package['evidence_count']}"
        )

        print(
            f"Response Ready: "
            f"{evidence_package['response_ready']}"
        )

        for evidence in (
            evidence_package.get(
                "evidence",
                [],
            )
        ):

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Rank: "
                f"{evidence.get('final_rank')}"
            )

            print(
                f"Policy: "
                f"{evidence.get('policy_id')} "
                f"v{evidence.get('version')}"
            )

            print(
                f"Section: "
                f"{evidence.get('section_title')}"
            )

            print(
                f"Pages: "
                f"{evidence.get('page_numbers')}"
            )

            content = (
                evidence.get(
                    "content"
                )
                or ""
            )

            print(
                content[:350]
            )

        # =========================================================
        # Print the final grounded response
        # =========================================================

        final_response = result[
            "response"
        ]

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL GROUNDED RESPONSE"
        )

        print(
            "=" * 70
        )

        print(
            final_response[
                "answer"
            ]
        )

        print(
            "\nCITATIONS"
        )

        for citation in final_response[
            "citations"
        ]:

            print(
                citation
            )

    except Exception as error:

        print(
            "\nOrchestrator execution failed."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error details: "
            f"{error}"
        )