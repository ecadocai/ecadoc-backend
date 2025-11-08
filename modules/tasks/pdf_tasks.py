"""
Celery tasks for asynchronous PDF processing.
"""
import os
from typing import Dict
from modules.tasks.celery_app import celery_app
from modules.pdf_processing.service import pdf_processor
from modules.database import db_manager
from modules.config.logger import logger


@celery_app.task(name="process_pdf_upload")
def process_pdf_upload(temp_path: str, filename: str, user_id: int) -> Dict[str, str]:
    try:
        # Mark as processing if job exists
        try:
            db_manager.update_job_status(celery_app.current_task.request.id, "processing")
        except Exception:
            pass
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temp file not found: {temp_path}")

        with open(temp_path, "rb") as f:
            file_content = f.read()

        result = pdf_processor.upload_and_index_pdf(file_content, filename, user_id)
        logger.info("pdf_upload_processed", extra={"user_id": user_id, "filename": filename})
        try:
            db_manager.update_job_status(
                celery_app.current_task.request.id,
                "completed",
                metadata={"doc_id": result.get("doc_id", ""), "filename": filename},
            )
        except Exception:
            pass
        return {"status": "completed", "doc_id": result.get("doc_id", "")}
    except Exception as e:
        logger.error("pdf_upload_failed", extra={"error": str(e), "filename": filename, "user_id": user_id})
        try:
            db_manager.update_job_status(
                celery_app.current_task.request.id, "failed", error=str(e)
            )
        except Exception:
            pass
        return {"status": "failed", "error": str(e)}
    finally:
        # cleanup temp file best-effort
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
