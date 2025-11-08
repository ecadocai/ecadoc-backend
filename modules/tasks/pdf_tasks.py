"""
Celery tasks for asynchronous PDF processing.
"""
import os
from typing import Dict
from modules.tasks.celery_app import celery_app
from modules.pdf_processing.service import pdf_processor
from modules.config.logger import logger


@celery_app.task(name="process_pdf_upload")
def process_pdf_upload(temp_path: str, filename: str, user_id: int) -> Dict[str, str]:
    try:
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temp file not found: {temp_path}")

        with open(temp_path, "rb") as f:
            file_content = f.read()

        result = pdf_processor.upload_and_index_pdf(file_content, filename, user_id)
        logger.info("pdf_upload_processed", extra={"user_id": user_id, "filename": filename})
        return {"status": "completed", "doc_id": result.get("doc_id", "")}
    except Exception as e:
        logger.error("pdf_upload_failed", extra={"error": str(e), "filename": filename, "user_id": user_id})
        return {"status": "failed", "error": str(e)}
    finally:
        # cleanup temp file best-effort
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

