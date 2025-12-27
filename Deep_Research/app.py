import sys
import os
import asyncio
import uuid
from typing import Dict, Optional, Any
from enum import Enum

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from service import run_deep_research


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobState(BaseModel):
    status: JobStatus
    result: Optional[str] = None
    error: Optional[str] = None


JOBS: Dict[str, JobState] = {}


class ResearchRequest(BaseModel):
    prompt: Optional[str] = None
    query: Optional[str] = None  # ICA sends 'query'
    job_id: Optional[str] = None

    class Config:
        extra = "allow"  # Allow extra fields like 'stream', 'context'


app = FastAPI(title="Deep Research API (Async Job Mode)")

# Add CORS for A2A / external UI access
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/debug/routes")
def debug_routes():
    return sorted([f"{','.join(r.methods or [])} {r.path}" for r in app.router.routes])

@app.get("/.well-known/agent-card.json")
@app.get("/agent-card.json")
def get_agent_card(request: Request):
    import json
    with open(os.path.join(os.path.dirname(__file__), "agent.json"), "r") as f:
        data = json.load(f)
    
    # Dynamically set the URL to the current server address
    # This ensures it works on Render, localhost, or anywhere else without hardcoding
    data["url"] = str(request.base_url).rstrip("/")
    return data


async def background_deep_research(job_id: str, prompt: str):
    print(f"🚀 Job {job_id} started. Prompt: {prompt[:50]}...")

    if job_id in JOBS:
        JOBS[job_id].status = JobStatus.RUNNING

    try:
        report = await run_deep_research(prompt)

        if job_id in JOBS:
            JOBS[job_id].status = JobStatus.DONE
            JOBS[job_id].result = report
            print(f"✅ Job {job_id} finished successfully.")

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        if job_id in JOBS:
            JOBS[job_id].status = JobStatus.FAILED
            JOBS[job_id].error = str(e)
            JOBS[job_id].result = f"Failed: {str(e)}"


def is_job_id_missing(job_id: Optional[str]) -> bool:
    if not job_id:
        return True
    cleaned = job_id.strip()
    if not cleaned:
        return True
    if cleaned.lower() == "no keywords added":
        return True
    return False


# <<< changed
def return_simple_message(text: str):
    """
    Return the structured response ICA docs specify:

    {
        "status": "success",
        "invocationId": "<uuid4>",
        "response": [{"message": text, "type": "text"}]
    }
    """
    invocation_id = str(uuid.uuid4())
    content = {
        "status": "success",
        "invocationId": invocation_id,
        "response": [
            {
                "message": text,
                "type": "text",
            }
        ],
    }
    print(f"DEBUG - ICA Response: {content}")
    return JSONResponse(content=content)
# <<< end changed


@app.get("/health")
def health_check():
    return {"ok": True}


@app.post("/deep-research")
async def deep_research_endpoint(request: Request):
    body = await request.json()
    print(f"DEBUG - ICA Request Body: {body}")

    raw_job_id = body.get("job_id")
    job_id_missing = is_job_id_missing(raw_job_id)

    # Case 1: start new job
    if job_id_missing:
        prompt = (body.get("prompt") or body.get("query") or "").strip()
        if not prompt:
            return return_simple_message(
                "Error: Please provide a prompt to start research."
            )

        new_job_id = str(uuid.uuid4())

        JOBS[new_job_id] = JobState(status=JobStatus.QUEUED, result=None)

        asyncio.create_task(background_deep_research(new_job_id, prompt))

        return return_simple_message(
            f"Research started. Job ID: {new_job_id}. "
            f"Re-run with this Job ID to get status/result."
        )

    # Case 2: check status
    job_id = raw_job_id.strip()

    if job_id not in JOBS:
        return return_simple_message(
            "Job ID not found. Start new research by leaving Job ID blank."
        )

    job = JOBS[job_id]

    if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
        return return_simple_message(
            f"Research in progress for Job ID {job_id}. "
            f"Please try again in 30-60 seconds."
        )

    if job.status == JobStatus.DONE:
        return return_simple_message(job.result)

    if job.status == JobStatus.FAILED:
        return return_simple_message(
            f"Research failed for Job ID {job_id}. Error: {job.result or job.error}"
        )

    return return_simple_message("Unknown state")


@app.post("/deep-research/sync")
async def deep_research_sync(request: Request):
    """
    Synchronous endpoint for tool calling.
    Waits for the research to complete and returns the result immediately.
    """
    body = await request.json()
    print(f"DEBUG - Sync Request Body: {body}")
    
    prompt = (body.get("prompt") or body.get("query") or "").strip()
    if not prompt:
        return return_simple_message("Error: Please provide a prompt.")

    try:
        # Run research and await result
        report = await run_deep_research(prompt)
        return return_simple_message(report)
    except Exception as e:
        print(f"❌ Sync Research Failed: {e}")
        return return_simple_message(f"Research failed: {str(e)}")



# ===== MCP SERVER SETUP =====
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types
import asyncio
import anyio
from typing import Dict, Any
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

mcp_server = Server("deep-research")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="deep_research",
            description="Performs deep research on a topic and returns a comprehensive report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The topic or question to research deeply",
                    }
                },
                "required": ["prompt"],
            },
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    if name != "deep_research":
        raise ValueError(f"Unknown tool: {name}")

    prompt = arguments.get("prompt")
    if not prompt:
        raise ValueError("Prompt is required")

    try:
        report = await run_deep_research(prompt)
        return [TextContent(type="text", text=report)]
    except Exception as e:
        print(f"Tool execution failed: {e}")
        return [TextContent(type="text", text=f"Error executing research: {str(e)}")]


# ===== MCP SSE ENDPOINTS (MANUAL WIRING) =====

# Session storage: Map session_id -> input_stream_sender (to send messages TO the server)
# The server writes responses to the output_stream, which we consume in /sse
web_server_sessions: Dict[str, anyio.streams.memory.MemoryObjectSendStream] = {}

@app.get("/sse")
async def handle_sse(request: Request):
    """
    Manual SSE endpoint for MCP.
    """
    # Create memory streams for bidirectional communication
    # client_input: /messages -> input_send -> input_recv -> mcp_server
    # server_output: mcp_server -> output_send -> output_recv -> /sse
    
    input_send, input_recv = anyio.create_memory_object_stream(100)
    output_send, output_recv = anyio.create_memory_object_stream(100)
    
    session_id = str(uuid.uuid4())
    web_server_sessions[session_id] = input_send
    
    print(f"Starting SSE session: {session_id}")

    async def sse_generator():
        # 1. Send the "endpoint" event with the session URL
        # The client uses this URl to post messages
        yield {
             "event": "endpoint", 
             "data": f"/messages?session_id={session_id}"
        }
        
        # 2. Start the MCP server loop in the background
        # It consumes from input_recv and writes to output_send
        server_task = asyncio.create_task(
            mcp_server.run(
                input_recv,
                output_send,
                mcp_server.create_initialization_options()
            )
        )
        
        try:
             # 3. Consume output from the server and yield as SSE
             async with output_recv:
                 async for message in output_recv:
                     # message is a JSON-RPC message object (or types.JSONRPCMessage)
                     # We verify it's a valid SSE message format
                     # Use standard server-sent events format: event: message\ndata: ...
                     
                     # Simple mcp serialization:
                     if isinstance(message, Exception):
                         yield {"event": "error", "data": str(message)}
                         continue
                         
                     # The mcp server run loop outputs fully formatted JSON-RPC messages/exceptions over the stream
                     # We simply wrap them in the 'message' event standard
                     
                     # Note: mcp_server.run writes the *object* to the stream.
                     # We need to serialize it to JSON string for SSE data.
                     # Or does it write bytes/strings?
                     # The SDK types say: run(read_stream: MemoryObjectReceiveStream[JSONRPCMessage | Exception], ...)
                     # So valid messages are objects. We must serialize.
                     
                     try:
                        json_str = message.model_dump_json() # Pydantic v2
                     except AttributeError:
                        json_str = message.json() # Pydantic v1 fallback
                     
                     yield {"event": "message", "data": json_str}

        except asyncio.CancelledError:
             print(f"SSE Client disconnected {session_id}")
        except Exception as e:
             print(f"SSE Generator Error: {e}")
             yield {"event": "error", "data": str(e)}
        finally:
             server_task.cancel()
             if session_id in web_server_sessions:
                 del web_server_sessions[session_id]
             try:
                 await server_task
             except asyncio.CancelledError:
                 pass

    return EventSourceResponse(sse_generator())


@app.post("/messages")
async def handle_messages(request: Request):
    """
    Forward client messages to the specific session's input stream.
    """
    session_id = request.query_params.get("session_id") or request.query_params.get("sessionId")
    
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "Missing session_id"})
        
    input_sender = web_server_sessions.get(session_id)
    if not input_sender:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
        
    try:
        body = await request.json()
        
        # Parse into MCP JSONRPCMessage
        # We need to identify correct type?
        # Actually, mcp_server.run expects input to be JSONRPCMessage objects.
        # We can use the Pydantic adapter or manually parse.
        # The easiest way is to use the type adapter from mcp.types
        
        import mcp.types as types
        # Basic deserialization - we can try generic dict if the server accepts it?
        # No, python mcp sdk is typed.
        # We need to convert the dict 'body' into a JSONRPCMessage.
        
        # Safe catch-all parsing using Pydantic Adapter for the Union type?
        # mcp.types.JSONRPCMessage is a Union.
        from pydantic import TypeAdapter
        message = TypeAdapter(types.JSONRPCMessage).validate_python(body)
        
        await input_sender.send(message)
        return JSONResponse(content={"status": "accepted"})
        
    except Exception as e:
        print(f"Error handling message: {e}")

