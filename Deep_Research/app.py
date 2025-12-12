import sys
import os

# Add src directory to path so deep_research package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, Request
from service import run_deep_research

app = FastAPI(title="Deep Research API")

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

    try:
        result_text = await run_deep_research(prompt)
        return {"result": result_text}
    except Exception as e:
        return {"result": f"Error processing request: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
