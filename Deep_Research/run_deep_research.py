import os
import asyncio
from dotenv import load_dotenv

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from deep_research.research_agent_full import deep_researcher_builder


def show_env_check():
    """Quick sanity check that OPENAI_API_KEY is visible."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY is NOT set. Check your .env file.")
    else:
        print("✅ OPENAI_API_KEY is set (not printing for security).")


async def main():
    show_env_check()

    # ----- Build the full multi-step agent -----
    print("🔧 Building deep research agent graph...")
    memory = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=memory)

    # ----- Your stakeholder analysis prompt -----
    user_prompt = "I need deep company analysis on Westpac Bank in Australia, hihglighting their key goals, challenges and performance. Focus on the latest information, reports and news as of 2025. Also do a deep research on Westpac Bank's technology stack and technology challenges and where they are already looking to invest or there is a need for them to invest, given the goals and challenges. Since you are an IBM consulting, you need to look for potential opportunities in IT Services to partner with Westpac Bank. Your key offerings are in the areas of ERP, core banking modernization, managed services, Ai integration services, mainframe modernization services, banking specific ISV partnership and solutions"

    # LangGraph config: increase recursion_limit enough for full pipeline
    thread_config = {
        "configurable": {
            "thread_id": "researchoutput",
            "recursion_limit": 50,
        }
    }

    # ----- Invoke the full graph -----
    print("⏳ Running deep research workflow...")
    result = await full_agent.ainvoke(
        {
            # These fields are expected by AgentInputState / AgentState in your graph
            "messages": [HumanMessage(content=user_prompt)],
            "user_request": user_prompt,
        },
        config=thread_config,
    )

    # ----- Extract the final profile -----
    final_profile = result.get("final_report")
    if not final_profile:
        # Fallback: try draft_report if final_report missing
        final_profile = result.get("draft_report")

    if not final_profile:
        print("⚠ No final_report or draft_report found in result state.")
        print("Full result keys:", list(result.keys()))
        return

    print("\n===== FINAL STAKEHOLDER PROFILE =====\n")
    print(final_profile)

    # Save to markdown
    output_path = "output.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_profile)

    print(f"\n💾 Saved profile to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
