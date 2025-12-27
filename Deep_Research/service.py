import os
import asyncio
from dotenv import load_dotenv

import sys
import os

# Add src directory to path so deep_research package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from deep_research.research_agent_full import deep_researcher_builder

from typing import Callable, Awaitable

async def run_deep_research(prompt: str, status_callback: Callable[[str], Awaitable[None]] = None) -> str:
    """
    Runs the deep research agent with the given prompt and returns the final report.
    accepts an optional status_callback to report progress.
    """
    load_dotenv()
    
    # Check for API key (optional but good for debugging)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment")

    # ----- Build the full multi-step agent -----
    memory = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=memory)

    # LangGraph config: increase recursion_limit enough for full pipeline
    thread_config = {
        "configurable": {
            "thread_id": "researchoutput",
            "recursion_limit": 50,
        }
    }

    # ----- Invoke the full graph with streaming -----
    print(f"⏳ Running deep research for prompt: {prompt[:50]}...")
    
    final_state = None
    
    # helper for safe callback
    async def report(msg):
        if status_callback:
            try:
                await status_callback(msg)
            except Exception:
                pass
        print(f"STATUS: {msg}")

    # Use astream_events to get real-time progress
    async for event in full_agent.astream_events(
        {
            "messages": [HumanMessage(content=prompt)],
            "user_request": prompt,
        },
        config=thread_config,
        version="v2"
    ):
        kind = event["event"]
        
        # Capture Tool Usage
        if kind == "on_tool_start":
            tool_name = event['name']
            if tool_name != "_Exception": # Ignore internal noise
                await report(f"Using tool: {tool_name}...")
        
        # Capture Chain/Node Transitions (High level steps)
        elif kind == "on_chain_start":
            node_name = event['name']
            if node_name and node_name not in ["LangGraph", "__start__", "FlatMap", "RunnableSequence"]:
                await report(f"Starting step: {node_name}...")
                
        # Capture Final Result from "on_chain_end" of the main graph
        elif kind == "on_chain_end" and event['name'] == "LangGraph":
             # The final output is in event['data']['output']
             final_state = event['data'].get('output')

    if not final_state:
        return "Error: No final state returned from graph stream."
    
    # ----- Extract the final profile -----
    final_profile = final_state.get("final_report")
    if not final_profile:
        # Fallback: try draft_report if final_report missing
        final_profile = final_state.get("draft_report")

    if not final_profile:
        return "Error: No final_report or draft_report found in result state."

    return final_profile

    # ----- Extract the final profile -----
    final_profile = result.get("final_report")
    if not final_profile:
        # Fallback: try draft_report if final_report missing
        final_profile = result.get("draft_report")

    if not final_profile:
        return "Error: No final_report or draft_report found in result state."

    return final_profile
