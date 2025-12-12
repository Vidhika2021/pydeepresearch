"""
Multi-agent supervisor for coordinating research across multiple specialized agents.

This module implements a supervisor pattern where:
1. A supervisor agent coordinates research activities and delegates tasks
2. Multiple researcher agents work on specific sub-topics independently
3. Results are aggregated and compressed for final reporting

For simplicity and reliability, this version runs a SINGLE
supervisor → tools pass (no internal recursion loop). LangGraph
handles this as a subgraph:
    START → supervisor → supervisor_tools → END
"""

import asyncio
from typing_extensions import Literal

from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompts import (
    lead_researcher_with_multiple_steps_diffusion_double_check_prompt,
)
from deep_research.research_agent import researcher_agent
from deep_research.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
)
from deep_research.utils import get_today_str, think_tool, refine_draft_report


# ===== Helper: extract notes from tool messages =====


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history."""
    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]


# ===== Optional: Jupyter compatibility =====

try:
    import nest_asyncio

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass
except ImportError:
    pass


# ===== CONFIGURATION =====

SUPERVISOR_TOOL_SPECS = [
    ConductResearch,
    ResearchComplete,
    think_tool,
    refine_draft_report,
]


def get_supervisor_model_with_tools():
    """Lazy-load supervisor model and bind tools AFTER env & API keys are set."""
    from langchain.chat_models import init_chat_model

    base_model = init_chat_model(model="openai:gpt-5")
    return base_model.bind_tools(SUPERVISOR_TOOL_SPECS)


# Not strictly used for looping anymore, but passed into the prompt
max_researcher_iterations = 15
max_concurrent_researchers = 3


# ===== SUPERVISOR NODES =====


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities (single step).

    Decides what to do next (call tools, continue, or finish).
    This node only runs ONCE per subgraph execution; all subsequent
    tool calls happen in supervisor_tools, then the subgraph ends.
    """
    raw_supervisor_messages = state.get("supervisor_messages", [])
    research_brief = state.get("research_brief", "")

    system_message = (
        lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
            date=get_today_str(),
            max_concurrent_research_units=max_concurrent_researchers,
            max_researcher_iterations=max_researcher_iterations,
        )
    )

    # When calling OpenAI:
    # - skip tool messages entirely
    # - skip assistant messages that already contain tool_calls
    filtered_history: list[BaseMessage] = []
    for m in raw_supervisor_messages:
        if isinstance(m, ToolMessage):
            continue
        # some earlier nodes may have put plain strings in supervisor_messages
        if isinstance(m, str):
            filtered_history.append(HumanMessage(content=m))
            continue
        if getattr(m, "tool_calls", None):
            continue
        filtered_history.append(m)

    # If no prior history got through, at least give it the research brief
    if not filtered_history and research_brief:
        filtered_history.append(HumanMessage(content=research_brief))

    messages = [SystemMessage(content=system_message)] + filtered_history

    supervisor_model_with_tools = get_supervisor_model_with_tools()
    response = await supervisor_model_with_tools.ainvoke(messages)

    # We append the new assistant response (with tool_calls) to the existing history;
    # supervisor_tools() will actually execute those tool calls.
    new_history = raw_supervisor_messages + [response]

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": new_history,
            # keep the old research_iterations if present; not used for looping now
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState,
) -> Command[Literal["__end__"]]:
    """Execute supervisor decisions (tool calls) and end the subgraph.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Optionally refining the draft report
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_brief = state.get("research_brief", "")
    draft_report = state.get("draft_report", "")

    if not supervisor_messages:
        # Safety guard
        return Command(
            goto="__end__",
            update={
                "notes": [],
                "research_brief": research_brief,
                "draft_report": draft_report,
            },
        )

    most_recent_message = supervisor_messages[-1]
    tool_calls = getattr(most_recent_message, "tool_calls", None) or []

    # If the model didn't ask for any tools, just end gracefully
    if not tool_calls:
        return Command(
            goto="__end__",
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": research_brief,
                "draft_report": draft_report,
            },
        )

    tool_messages: list[ToolMessage] = []
    all_raw_notes: list[str] = []

    try:
        think_tool_calls = [tc for tc in tool_calls if tc["name"] == "think_tool"]
        conduct_research_calls = [
            tc for tc in tool_calls if tc["name"] == "ConductResearch"
        ]
        refine_report_calls = [
            tc for tc in tool_calls if tc["name"] == "refine_draft_report"
        ]

        # 1) think_tool (sync)
        for tc in think_tool_calls:
            observation = think_tool.invoke(tc["args"])
            tool_messages.append(
                ToolMessage(
                    content=observation,
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )

        # 2) ConductResearch (async, parallel)
        if conduct_research_calls:
            coros = [
                researcher_agent.ainvoke(
                    {
                        "researcher_messages": [
                            HumanMessage(content=tc["args"]["research_topic"])
                        ],
                        "research_topic": tc["args"]["research_topic"],
                    }
                )
                for tc in conduct_research_calls
            ]
            results = await asyncio.gather(*coros)

            for res, tc in zip(results, conduct_research_calls):
                tool_messages.append(
                    ToolMessage(
                        content=res.get(
                            "compressed_research",
                            "Error synthesizing research report",
                        ),
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )
                all_raw_notes.append("\n".join(res.get("raw_notes", [])))

        # 3) refine_draft_report (sync)
        if refine_report_calls:
            notes = get_notes_from_tool_calls(supervisor_messages) + all_raw_notes
            findings = "\n".join(notes)

            draft_report = refine_draft_report.invoke(
                {
                    "research_brief": research_brief,
                    "findings": findings,
                    "draft_report": draft_report,
                }
            )

            for tc in refine_report_calls:
                tool_messages.append(
                    ToolMessage(
                        content=draft_report,
                        name=tc["name"],
                        tool_call_id=tc["id"],
                    )
                )

    except Exception:
        # On error, bail out cleanly
        return Command(
            goto="__end__",
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": research_brief,
                "draft_report": draft_report,
            },
        )

    # Final state returned back to the outer graph
    final_notes = get_notes_from_tool_calls(supervisor_messages + tool_messages)

    return Command(
        goto="__end__",
        update={
            "supervisor_messages": supervisor_messages + tool_messages,
            "raw_notes": state.get("raw_notes", []) + all_raw_notes,
            "notes": final_notes,
            "research_brief": research_brief,
            "draft_report": draft_report,
        },
    )


# ===== GRAPH CONSTRUCTION =====

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# One pass: START → supervisor → supervisor_tools → END
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("supervisor_tools", END)

supervisor_agent = supervisor_builder.compile()
