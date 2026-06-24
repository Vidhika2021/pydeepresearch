"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

import time
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from deep_research.state_research import Summary
from deep_research.prompts import (
    summarize_webpage_prompt,
    report_generation_with_draft_insight_prompt,
)

# ===== UTILITY FUNCTIONS =====


def get_today_str() -> str:
    """Return current date in a Windows-safe, human-readable format."""
    dt = datetime.now()
    weekday = dt.strftime("%a")
    month = dt.strftime("%b")
    day = dt.day
    year = dt.year
    return f"{weekday} {month} {day}, {year}"


def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()


# ===== CONFIGURATION (lazy-loaded models & clients) =====


class RetryingChatModel:
    """A wrapper around LangChain chat models to automatically retry on transient API/gateway errors."""
    def __init__(self, model):
        self.model = model

    def invoke(self, *args, **kwargs):
        import time
        for attempt in range(1, 5):
            try:
                return self.model.invoke(*args, **kwargs)
            except Exception as e:
                is_transient = "Model not found" in str(e) or "rate limit" in str(e).lower() or "timeout" in str(e).lower() or "50" in str(e)
                if is_transient and attempt < 4:
                    print(f"Transient model error in invoke (attempt {attempt}/4): {e}. Retrying in 2s...")
                    time.sleep(2)
                else:
                    raise e

    async def ainvoke(self, *args, **kwargs):
        import asyncio
        for attempt in range(1, 5):
            try:
                return await self.model.ainvoke(*args, **kwargs)
            except Exception as e:
                is_transient = "Model not found" in str(e) or "rate limit" in str(e).lower() or "timeout" in str(e).lower() or "50" in str(e)
                if is_transient and attempt < 4:
                    print(f"Transient model error in ainvoke (attempt {attempt}/4): {e}. Retrying in 2s...")
                    await asyncio.sleep(2)
                else:
                    raise e

    def with_structured_output(self, schema, **kwargs):
        structured_model = self.model.with_structured_output(schema, **kwargs)
        return RetryingChatModel(structured_model)

    def bind_tools(self, tools, **kwargs):
        bound_model = self.model.bind_tools(tools, **kwargs)
        return RetryingChatModel(bound_model)

    def __getattr__(self, name):
        return getattr(self.model, name)


def get_chat_model(model: str = "gpt-4o", max_tokens: int = None):
    """Lazy-load chat model with custom environment/ICA overrides."""
    import os
    from langchain.chat_models import init_chat_model

    # Enforce gpt-4o strictly as requested by the user
    model_name = "gpt-4o"

    api_key = os.getenv("OPENAI_API_KEY", "sk-1b582adabcc54fae8ca0ba08463dc26a")
    base_url = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", "https://api.nextgen-beta.ica.ibm.com/ica/v1/chat-models"))

    kwargs = {
        "model": model_name,
        "model_provider": "openai",
        "api_key": api_key,
        "base_url": base_url,
    }
    if max_tokens is not None:
        # Cap max_tokens to 16384 as Azure/litellm endpoint limits it for gpt-4o
        kwargs["max_tokens"] = min(max_tokens, 16384)

    raw_model = init_chat_model(**kwargs)
    return RetryingChatModel(raw_model)


def get_summarization_model():
    """Lazy-load summarization model AFTER environment variables are loaded."""
    return get_chat_model(model="gpt-4o")


def get_writer_model():
    """Lazy-load writer model AFTER environment variables are loaded."""
    return get_chat_model(model="gpt-4o", max_tokens=8192)


def get_tavily_client():
    """Lazy-load Tavily client AFTER TAVILY_API_KEY is available."""
    from tavily import TavilyClient

    return TavilyClient()


MAX_CONTEXT_LENGTH = 250000

# ===== SEARCH FUNCTIONS =====


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    time_range: Optional[str] = None,
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content
        time_range: Time range to filter results by ('day', 'week', 'month', 'year')

    Returns:
        List of search result dictionaries
    """
    client = get_tavily_client()

    # Execute searches sequentially. Note: you can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        result = client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
            time_range=time_range,
        )
        search_docs.append(result)

    return search_docs


def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    # Retry loop to handle transient nextgen/ICA gateway errors
    structured_model = get_summarization_model().with_structured_output(Summary)
    summary = None
    for attempt in range(1, 4):
        try:
            summary = structured_model.invoke(
                [
                    HumanMessage(
                        content=summarize_webpage_prompt.format(
                            webpage_content=webpage_content, date=get_today_str()
                        )
                    )
                ]
            )
            break
        except Exception as e:
            print(f"Webpage summarization attempt {attempt} failed: {e}")
            if attempt == 3:
                print(f"Failed to summarize webpage after 3 attempts: {str(e)}")
                return (
                    webpage_content[:1000] + "..."
                    if len(webpage_content) > 1000
                    else webpage_content
                )
            time.sleep(2)

    # Format summary with clear structure
    formatted_summary = (
        f"<summary>\n{summary.summary}\n</summary>\n\n"
        f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
    )

    return formatted_summary


def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}

    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

    return unique_results


def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}

    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result["content"]
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(
                result["raw_content"][:MAX_CONTEXT_LENGTH]
            )

        summarized_results[url] = {"title": result["title"], "content": content}

    return summarized_results


def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output


# ===== RESEARCH TOOLS =====


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
    time_range: Optional[Literal["day", "week", "month", "year"]] = None,
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')
        time_range: Optional time range to filter results by ('day', 'week', 'month', 'year')

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        time_range=time_range,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool(parse_docstring=True)
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg],
    findings: Annotated[str, InjectedToolArg],
    draft_report: Annotated[str, InjectedToolArg],
):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: user's research request
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request

    Returns:
        refined draft report
    """

    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str(),
    )

    writer_model = get_writer_model()
    
    # Retry loop to handle transient nextgen/ICA gateway errors
    draft_report_msg = None
    for attempt in range(1, 4):
        try:
            draft_report_msg = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
            break
        except Exception as e:
            print(f"Refining draft report attempt {attempt} failed: {e}")
            if attempt == 3:
                raise e
            time.sleep(2)

    return draft_report_msg.content


@tool(parse_docstring=True)
def fetch_pdf_content(pdf_url: str) -> str:
    """Download a PDF file from a URL and extract its text content.

    This tool is critical for extracting raw, precise numbers, dates, tables, and statistics
    from official financial report PDFs or regulatory filings (such as ASX, SEC, or annual reports).

    Args:
        pdf_url: The direct URL to the PDF file to download and parse.

    Returns:
        A structured string containing the extracted text from the PDF pages, or an error message.
    """
    import io
    import urllib.request
    import ssl
    from pypdf import PdfReader

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        req = urllib.request.Request(pdf_url, headers=headers)
        # Timeout after 25 seconds to prevent hanging
        # Use unverified SSL context to bypass proxy/certificate verification errors
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=25, context=context) as response:
            pdf_data = response.read()
    except Exception as e:
        return f"Error downloading PDF from {pdf_url}: {e}"

    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        num_pages = len(reader.pages)
        
        extracted_text = []
        max_pages_to_extract = 10
        pages_to_read = min(num_pages, max_pages_to_extract)
        
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text(extraction_mode="layout") or ""
            extracted_text.append(f"--- Page {i+1} ---\n{page_text.strip()}\n")
            
        result = "\n".join(extracted_text)
        if num_pages > max_pages_to_extract:
            result += f"\n\n[TRUNCATED: PDF has {num_pages} pages total. Only the first {max_pages_to_extract} pages were extracted to prevent context length limits. Use more specific search queries if you need information from later pages.]"
            
        return result
    except Exception as e:
        return f"Error parsing PDF content: {e}"

