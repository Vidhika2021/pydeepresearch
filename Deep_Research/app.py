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
        logger.info(f"Running deep research with prompt: {prompt}")
        
        # ICA Timeout Protection: Wrap execution in 50s timeout
        # If research takes > 50s, return a partial/status message instead of crashing
        try:
            result = await asyncio.wait_for(run_deep_research(prompt), timeout=50.0)
        except asyncio.TimeoutError:
            logger.warning("Deep research timed out after 50s")
            return return_simple_message("Research is taking longer than expected. It is still running in the background, but we are returning this message to prevent a timeout. Please refine your query or try a more specific topic.")
        
        logger.info("Deep research completed")
        return return_simple_message(result)
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


# Global Job Tracking
RESEARCH_JOBS: Dict[str, Dict[str, Any]] = {}


async def run_research_task(job_id: str, prompt: str):
    """
    Background task wrapper to update job status.
    """
    RESEARCH_JOBS[job_id] = {"status": "running", "result": None, "logs": []}
    
    async def log_callback(msg: str):
        # Keep only last 10 logs to save memory, or just append
        RESEARCH_JOBS[job_id]["logs"].append(msg)
    
    try:
        print(f"DEBUG: Job {job_id} started via background task")
        # Pass callback to service
        result = await run_deep_research(prompt, status_callback=log_callback)
        RESEARCH_JOBS[job_id]["status"] = "completed"
        RESEARCH_JOBS[job_id]["result"] = result
        print(f"DEBUG: Job {job_id} completed successfully")
    except Exception as e:
        print(f"DEBUG: Job {job_id} failed: {e}")
        RESEARCH_JOBS[job_id]["status"] = "failed"
        RESEARCH_JOBS[job_id]["error"] = str(e)

mcp_server = Server("deep-research")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ping",
            description="A simple health check tool to verify MCP connection.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="deep_research",
            description="Performs deep research on a topic. NOTE: If research takes time, this tool returns a Job ID. You MUST then immediately call `get_research_status` repeatedly until the final report is retrieved.",
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
        ),
        Tool(
            name="get_research_status",
            description="Checks the status of a background research job. Returns logs if running.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Job ID returned by deep_research",
                    }
                },
                "required": ["job_id"],
            },
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    if name == "ping":
        return [TextContent(type="text", text="pong")]

    if name == "get_research_status":
        job_id = arguments.get("job_id")
        if not job_id:
            return [TextContent(type="text", text="Error: job_id is required")]
            
        job = RESEARCH_JOBS.get(job_id)
        if not job:
            return [TextContent(type="text", text="Job not found")]
            
        if job["status"] == "running":
            # Get latest log
            logs = job.get("logs", [])
            latest_log = logs[-1] if logs else "Processing..."
            return [TextContent(type="text", text=f"Job {job_id} is still running.\nLATEST STATUS: {latest_log}\n\nPlease check again in 10 seconds.")]
        elif job["status"] == "failed":
             return [TextContent(type="text", text=f"Job {job_id} failed: {job.get('error')}")]
        else:
             # Completed
             return [TextContent(type="text", text=job["result"])]


    if name == "deep_research":
        prompt = arguments.get("prompt")
        if not prompt:
             raise ValueError("Prompt is required")

        # Generate Job ID
        import uuid
        job_id = str(uuid.uuid4())[:8]
        
        # Start background task
        # We use asyncio.create_task to run it independently of this request
        task = asyncio.create_task(run_research_task(job_id, prompt))
        
        # Try to wait for it for 5s (Immediate Async Mode)
        try:
            print(f"DEBUG: Waiting for Job {job_id} (timeout 5s)")
            # Wait for the SPECIFIC task
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            
            # If we get here, task is done
            # Retrieve result from storage
            job = RESEARCH_JOBS.get(job_id)
            if job and job["status"] == "completed":
                return [TextContent(type="text", text=job["result"])]
            elif job and job["status"] == "failed":
                return [TextContent(type="text", text=f"Error: {job.get('error')}")]
            else:
                 return [TextContent(type="text", text="Error: Job finished but no result state found.")]
                 
        except asyncio.TimeoutError:
            print(f"DEBUG: Job {job_id} timed out (sync compliance). Returning Async Job ID.")
            # Do NOT cancel the task. Let it run.
            import json
            output_data = {
                "job_id": job_id,
                "status": "running",
                "instruction": f"SYSTEM INSTRUCTION: The research is continuing in the background. You MUST now call the tool `get_research_status` with job_id='{job_id}' to retrieve the progress. Repeat this every 10 seconds until the status is 'completed'."
            }
            return [TextContent(
                type="text", 
                text=json.dumps(output_data, indent=2)
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error launching research: {str(e)}")]

    raise ValueError(f"Unknown tool: {name}")




# ===== MCP SSE ENDPOINTS (MANUAL WIRING) =====

# Session storage: Map session_id -> input_stream_sender (to send messages TO the server)
# The server writes responses to the output_stream, which we consume in /sse
web_server_sessions: Dict[str, anyio.streams.memory.MemoryObjectSendStream] = {}

@app.get("/sse")
async def handle_sse(request: Request):
    """
    Manual SSE endpoint for MCP.
    """
    input_send, input_recv = anyio.create_memory_object_stream(100)
    output_send, output_recv = anyio.create_memory_object_stream(100)
    
    # Use 'sessionId' camelCase for ICA compatibility
    session_id = str(uuid.uuid4())
    web_server_sessions[session_id] = input_send
    
    print(f"Starting SSE session: {session_id}")

    async def sse_generator():
        # 1. Send the "endpoint" event with the ABSOLUTE session URL
        # ICA requires absolute URL and camelCase sessionId
        base_url = str(request.base_url).rstrip("/")
        endpoint_url = f"{base_url}/messages?sessionId={session_id}"
        
        yield {
             "event": "endpoint", 
             "data": endpoint_url
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
                 async for session_message in output_recv:
                     # The mcp server yields SessionMessage objects or Exceptions
                     
                     if isinstance(session_message, Exception):
                         yield {"event": "error", "data": str(session_message)}
                         continue
                     
                     # Unwrap SessionMessage to get the actual JSONRPCMessage
                     # We must verify if it's a SessionMessage or already unwrapped (defensive)
                     if hasattr(session_message, "message"):
                         message_obj = session_message.message
                     else:
                         message_obj = session_message
                     
                     # Serialize JSON-RPC message to string
                     try:
                        json_str = message_obj.model_dump_json(by_alias=True, exclude_none=True)
                     except AttributeError:
                        json_str = message_obj.json(by_alias=True, exclude_none=True)
                     
                     # ICA/Spec expects event: message
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
    # 1. Robust Session ID extraction
    # Priority: Query Param (sessionId > session_id) > Header (mcp-session-id)
    session_id = request.query_params.get("sessionId") or request.query_params.get("session_id")
    
    if not session_id:
        session_id = request.headers.get("mcp-session-id")
        
    if not session_id:
        return JSONResponse(status_code=400, content={"error": "Missing sessionId parameter"})
        
    input_sender = web_server_sessions.get(session_id)
    if not input_sender:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
        
    try:
        body = await request.json()
        
        # Parse into MCP JSONRPCMessage
        import mcp.types as types
        from pydantic import TypeAdapter
        from mcp.shared.message import SessionMessage
        
        message = TypeAdapter(types.JSONRPCMessage).validate_python(body)
        
        # Wrap in SessionMessage as expected by mcp.server.Server._receive_loop
        # (Verified: SDK expects MemoryObjectReceiveStream[SessionMessage | Exception])
        session_message = SessionMessage(message=message)
        
        await input_sender.send(session_message)
        return JSONResponse(content={"status": "accepted"})
        
    except Exception as e:
        print(f"Error handling message: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

