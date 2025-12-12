import sys
import os

# Add src directory to path so deep_research package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service import run_deep_research

app = FastAPI(title="Deep Research API")

class ResearchRequest(BaseModel):
    prompt: str

@app.post("/deep-research")
async def deep_research_endpoint(request: ResearchRequest):
    try:
        report = await run_deep_research(request.prompt)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
