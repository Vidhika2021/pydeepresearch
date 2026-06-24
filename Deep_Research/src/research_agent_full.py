"""
Full Multi-Agent Research System

This module integrates all components of the research system:
- User clarification and scoping
- Research brief generation
- Multi-agent research coordination
- Final report generation

The system orchestrates the complete research workflow from initial user
input through final report delivery.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str, get_writer_model
from deep_research.prompts import (
    final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt,
    report_verification_prompt,
)
from deep_research.state_scope import AgentState, AgentInputState
from deep_research.research_agent_scope import (
    clarify_with_user,
    write_research_brief,
    write_draft_report,
)
from deep_research.multi_agent_supervisor import supervisor_agent

# ===== Config =====


# ===== FINAL REPORT GENERATION =====


async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("user_request", ""),
    )

    writer_model = get_writer_model()
    
    # Retry loop for final report generation to handle transient gateway errors
    final_report = None
    for attempt in range(1, 4):
        try:
            final_report = await writer_model.ainvoke(
                [HumanMessage(content=final_report_prompt)]
            )
            break
        except Exception as e:
            print(f"Final report generation attempt {attempt} failed: {e}")
            if attempt == 3:
                raise e
            import asyncio
            await asyncio.sleep(2)

    # Verification and correction pass
    verification_prompt = report_verification_prompt.format(
        findings=findings,
        report=final_report.content,
    )
    
    corrected_report = final_report.content
    for attempt in range(1, 4):
        try:
            corrected_msg = await writer_model.ainvoke(
                [HumanMessage(content=verification_prompt)]
            )
            corrected_report = corrected_msg.content
            break
        except Exception as e:
            print(f"Report verification attempt {attempt} failed: {e}")
            if attempt == 3:
                print("Failed report verification after 3 attempts, falling back to original final report.")
            else:
                import asyncio
                await asyncio.sleep(2)

    return {
        "final_report": corrected_report,
        "messages": ["Here is the final report: " + corrected_report],
    }


# ===== GRAPH CONSTRUCTION =====

# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the full workflow
agent = deep_researcher_builder.compile()
