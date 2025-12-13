import sys
import os
import asyncio
import json

# Add src directory to path so deep_research package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from service import run_deep_research

app = FastAPI(title="Deep Research API")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Deep Research API is running"}

@app.post("/deep-research")
async def deep_research_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Try all common shapes tools may send
    prompt = (
        body.get("prompt")
        or body.get("input", {}).get("prompt")
        or body.get("inputs", {}).get("prompt")
        or body.get("parameters", {}).get("prompt")
    )

    if not prompt:
        return {"result": f"ERROR: No prompt received. Body keys: {list(body.keys())}"}

    print("🔥 Prompt received:", prompt)

    async def event_generator():
        # Start the research as a background task
        task = asyncio.create_task(run_deep_research(prompt))
        
        # Loop until the task is complete
        while not task.done():
            # Wait for either completion or timeout (heartbeat interval)
            done, pending = await asyncio.wait([task], timeout=8.0)
            
            if task in pending:
                # Task is still running, send a heartbeat
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Still researching...'})}\n\n"
            else:
                # Task completed, loop will break naturally
                break
        
        # Determine the result or error
        try:
            result_text = await task
            # Send final result
            yield f"data: {json.dumps({'status': 'completed', 'result': result_text})}\n\n"
        except Exception as e:
            # Send error
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
        
        # End stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
