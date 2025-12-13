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

def is_job_id_missing(job_id: Optional[str]) -> bool:
    """
    Checks if job_id matches any of the 'missing' criteria for ICA:
    - None
    - Empty string
    - Whitespace only
    - 'No keywords added'
    """
    if not job_id:
        return True
    cleaned = job_id.strip()
    if not cleaned:
        return True
    if cleaned.lower() == "no keywords added":
        return True
    return False

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"ok": True}

@app.post("/deep-research")
async def deep_research_endpoint(request: ResearchRequest):
    # Normalize inputs
    # ICA specific: treat "No keywords added" as invalid job_id
    raw_job_id = request.job_id
    job_id_missing = is_job_id_missing(raw_job_id)

    # --- Case 1: Start New Job (Missing job_id) ---
    if job_id_missing:
        # Validate prompt only when starting new job
        prompt = (request.prompt or "").strip()
        if not prompt:
             return {"result": "Error: Please provide a prompt to start research."}

        new_job_id = str(uuid.uuid4())
        
        # Initialize state
        JOBS[new_job_id] = JobState(status=JobStatus.QUEUED, result=None)
        
        # Start background task
        asyncio.create_task(background_deep_research(new_job_id, prompt))
        
        # Return single-line success message
        return {
            "result": f"Research started. Job ID: {new_job_id}. Re-run with this Job ID to get status/result."
        }

    # --- Case 2: Check Status (Job ID Provided) ---
    job_id = raw_job_id.strip()

    if job_id not in JOBS:
        return {
            "result": "Job ID not found. Start new research by leaving Job ID blank."
        }
    
    job = JOBS[job_id]

    if job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
        return {
            "result": f"Research in progress for Job ID {job_id}. Please try again in 30-60 seconds."
        }
    
    if job.status == JobStatus.DONE:
        # Return the actual full report (can be multi-line as it's the final result)
        return {
            "result": "Done"
        }
    
    if job.status == JobStatus.FAILED:
        return {
            "result": f"Research failed for Job ID {job_id}. Error: {job.result or job.error}"
        }
    
    return {"result": "Unknown state"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
