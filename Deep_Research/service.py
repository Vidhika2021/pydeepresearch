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

async def run_deep_research(prompt: str) -> str:
    """
    Runs the deep research agent with the given prompt and returns the final report.
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

    # ----- Invoke the full graph -----
    print(f"⏳ Running deep research for prompt: {prompt[:50]}...")
    result = await full_agent.ainvoke(
        {
            "messages": [HumanMessage(content=prompt)],
            "user_request": prompt,
        },
        config=thread_config,
    )

    # ----- Extract the final profile -----
    final_profile = result.get("final_report")
    if not final_profile:
        # Fallback: try draft_report if final_report missing
        final_profile = result.get("draft_report")

    if not final_profile:
        return "Error: No final_report or draft_report found in result state."

    return final_profile
