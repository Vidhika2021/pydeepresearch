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
    prompt: Optional[str] = None
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

    # Case 0: Empty prompt and no job ID
    if not prompt and not job_id:
        return {"result": "Error: Please provide a prompt to start research, or a Job ID to check status."}

    # Case 1: Start a new job (No job_id provided)
    if not job_id:
        new_job_id = str(uuid.uuid4())
        
        # Initialize state
        JOBS[new_job_id] = JobState(status=JobStatus.QUEUED, result=None)
        
        # Start background task
        asyncio.create_task(background_deep_research(new_job_id, prompt))
        
        # Return only "result" field as requested
        return {
            "result": f"Research started. Job ID: {new_job_id}\nPlease copy this Job ID and paste it into the 'job_id' field to check progress."
        }

    # Case 2: Check existing job status
    if job_id not in JOBS:
        return {
            "result": f"Error: Job ID '{job_id}' not found. Please leave Job ID blank to start a new search."
        }
    
    job = JOBS[job_id]

    if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
        return {
            "result": f"Research in progress for Job ID: {job_id}\nStatus: {job.status.value.upper()}.\nPlease try again in 30-60 seconds."
        }
    
    if job.status == JobStatus.DONE:
        # Return the actual full report
        return {
            "result": job.result
        }
    
    if job.status == JobStatus.FAILED:
        return {
            "result": f"Research terminated with error. Job ID: {job_id}\nDetails: {job.result or job.error}"
        }
    
    return {"result": "Unknown state"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
