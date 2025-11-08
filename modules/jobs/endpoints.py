"""
Jobs status endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from modules.database import db_manager
from modules.auth.deps import get_current_user_id

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("/{job_id}")
def get_job_status(job_id: str, user_id: int = Depends(get_current_user_id)):
    job = db_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return job

