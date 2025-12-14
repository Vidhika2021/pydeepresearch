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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
