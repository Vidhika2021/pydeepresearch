import sys
import os
import asyncio
import uuid
from typing import Dict, Optional, Any
from enum import Enum

# Add src directory to path so deep_research package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from service import run_deep_research

# --- Models & State ---

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class JobState(BaseModel):
    status: JobStatus
    result: Optional[str] = None
    error: Optional[str] = None

# Global in-memory job store
JOBS: Dict[str, JobState] = {}

class ResearchRequest(BaseModel):
    prompt: str
    job_id: Optional[str] = None

app = FastAPI(title="Deep Research API (Async Job Mode)")

# --- Helper Functions ---

async def background_deep_research(job_id: str, prompt: str):
    """
    Wrapper to run the actual research and update the global job store.
    """
    print(f"🚀 Job {job_id} started. Prompt: {prompt[:50]}...")
    
    # Update status to running
    if job_id in JOBS:
        JOBS[job_id].status = JobStatus.RUNNING

    try:
        # Calls the existing service function
        report = await run_deep_research(prompt)
        
        # Update success
        if job_id in JOBS:
            JOBS[job_id].status = JobStatus.DONE
            JOBS[job_id].result = report
            print(f"✅ Job {job_id} finished successfully.")

    except Exception as e:
        # Update failure
        print(f"❌ Job {job_id} failed: {e}")
        if job_id in JOBS:
            JOBS[job_id].status = JobStatus.FAILED
            JOBS[job_id].error = str(e)
            JOBS[job_id].result = f"Failed: {str(e)}" 

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"ok": True}

@app.post("/deep-research")
async def deep_research_endpoint(request: ResearchRequest):
    # Normalize inputs
    prompt = (request.prompt or "").strip()
    job_id = (request.job_id or "").strip()

    if not prompt:
         return {
            "status": "failed",
            "job_id": job_id if job_id else "nan",
            "result": "Failed: Prompt cannot be empty."
        }

    # Case 1: Start a new job (No job_id provided)
    if not job_id:
        new_job_id = str(uuid.uuid4())
        
        # Initialize state
        JOBS[new_job_id] = JobState(status=JobStatus.QUEUED, result=None)
        
        # Start background task using asyncio.create_task as requested
        # (FastAPI BackgroundTasks is also an option, but asyncio.create_task is explicit)
        asyncio.create_task(background_deep_research(new_job_id, prompt))
        
        return {
            "status": "queued",
            "job_id": new_job_id,
            "result": "Started. Re-run with this Job ID to fetch the result."
        }

    # Case 2: Check existing job status
    if job_id not in JOBS:
        return {
            "status": "not_found",
            "job_id": job_id,
            "result": "Job ID not found. Start new by leaving Job ID blank."
        }
    
    job = JOBS[job_id]

    if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
        return {
            "status": "running",
            "job_id": job_id,
            "result": "Still running. Re-run with the same Job ID in ~30–60 seconds."
        }
    
    if job.status == JobStatus.DONE:
        return {
            "status": "done",
            "job_id": job_id,
            "result": job.result
        }
    
    if job.status == JobStatus.FAILED:
        return {
            "status": "failed",
            "job_id": job_id,
            "result": job.result or f"Failed: {job.error}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
