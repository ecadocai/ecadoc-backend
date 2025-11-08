"""
PDF processing API endpoints for the Floor Plan Agent API
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List
from modules.pdf_processing.service import pdf_processor
from modules.auth.deps import get_current_user_id
from modules.tasks.pdf_tasks import process_pdf_upload
from modules.config.logger import logger
import os
import tempfile

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
async def upload_pdf(
    files: List[UploadFile] = File(...),
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Upload and index PDF document(s) - supports both single and multiple files, requires JWT
    """
    try:
        logger.debug("upload_received", extra={"files": len(files)})
        
        # Validate that we have files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Filter out empty files (common when no file is selected)
        valid_files = [f for f in files if f.filename and f.filename.strip() != '']
        
        if len(valid_files) == 0:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        logger.debug("upload_validated", extra={"valid_files": len(valid_files)})
        
        # Validate file types
        for file in valid_files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        # Handle single file (backward compatibility)
        if len(valid_files) == 1:
            file = valid_files[0]
            content = await file.read()
            os.makedirs("/tmp/ecadoc_uploads", exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix="ecadoc_", dir="/tmp/ecadoc_uploads")
            with os.fdopen(fd, "wb") as out:
                out.write(content)
            task = process_pdf_upload.apply_async(args=[temp_path, file.filename, current_user_id])
            return {"status": "queued", "job_id": task.id}
        
        # Handle multiple files
        else:
            file_contents = []
            filenames = []
            
            jobs = []
            for file in valid_files:
                content = await file.read()
                os.makedirs("/tmp/ecadoc_uploads", exist_ok=True)
                fd, temp_path = tempfile.mkstemp(prefix="ecadoc_", dir="/tmp/ecadoc_uploads")
                with os.fdopen(fd, "wb") as out:
                    out.write(content)
                task = process_pdf_upload.apply_async(args=[temp_path, file.filename, current_user_id])
                jobs.append({"filename": file.filename, "job_id": task.id})
            return {"status": "queued", "jobs": jobs}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("upload_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload-multiple")
async def upload_multiple_pdfs(
    files: List[UploadFile] = File(...),
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Upload and index multiple PDF documents, requires JWT
    """
    try:
        logger.debug("upload_received", extra={"files": len(files)})
        
        # Validate that we have files
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Check if any file has empty filename (common when no file is selected)
        valid_files = [f for f in files if f.filename and f.filename.strip() != '']
        
        if len(valid_files) == 0:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        logger.debug("upload_validated", extra={"valid_files": len(valid_files)})
        
        # Validate file types
        for file in valid_files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        jobs = []
        for file in valid_files:
            content = await file.read()
            os.makedirs("/tmp/ecadoc_uploads", exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix="ecadoc_", dir="/tmp/ecadoc_uploads")
            with os.fdopen(fd, "wb") as out:
                out.write(content)
            task = process_pdf_upload.apply_async(args=[temp_path, file.filename, current_user_id])
            jobs.append({"filename": file.filename, "job_id": task.id})
        return {"status": "queued", "jobs": jobs}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("upload_error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/")
def list_documents(user_id: int | None = None, current_user_id: int = Depends(get_current_user_id)):
    """
    List all uploaded documents, optionally filtered by user
    """
    resolved_user_id = user_id if user_id is not None else current_user_id
    return pdf_processor.list_documents(resolved_user_id)

@router.get("/{doc_id}")
def get_document_info(doc_id: str):
    """
    Get information about a specific document
    """
    try:
        return pdf_processor.get_document_info(doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{doc_id}/pages")
def get_document_pages(doc_id: str):
    """
    Get page count for a document
    """
    try:
        info = pdf_processor.get_document_info(doc_id)
        return {"doc_id": doc_id, "pages": info["pages"]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{doc_id}/query")
async def query_document(doc_id: str, question: str = Form(...), k: int = Form(5)):
    """
    Query a document using RAG (Retrieval Augmented Generation)
    """
    try:
        result = pdf_processor.query_document(doc_id, question, k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{doc_id}")
def delete_document(doc_id: str, user_id: int | None = None, current_user_id: int = Depends(get_current_user_id)):
    """
    Delete a document and all associated files (PDF and vectors)
    Optionally verify user ownership
    """
    try:
        # If user_id is provided, verify ownership
        if user_id is not None:
            # Ensure the caller is the authenticated user when passing user_id explicitly
            if user_id != current_user_id:
                raise HTTPException(status_code=403, detail="Access denied: caller mismatch")
            doc_info = pdf_processor.get_document_info(doc_id)
            if doc_info.get("user_id") != current_user_id:
                raise HTTPException(status_code=403, detail="Access denied: You don't own this document")
        
        success = pdf_processor.delete_document_files(doc_id)
        if success:
            return {"message": "Document deleted successfully", "doc_id": doc_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document completely")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/user/{user_id}")
def delete_user_documents(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    """
    Delete all documents for a specific user
    """
    try:
        if user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied to requested user's documents")

        documents = pdf_processor.list_documents(user_id)
        deleted_count = 0
        failed_deletions = []
        
        for doc_id, doc_info in documents.items():
            try:
                success = pdf_processor.delete_document_files(doc_id)
                if success:
                    deleted_count += 1
                else:
                    failed_deletions.append({"doc_id": doc_id, "filename": doc_info["filename"]})
            except Exception as e:
                failed_deletions.append({"doc_id": doc_id, "filename": doc_info["filename"], "error": str(e)})
        
        return {
            "message": f"Deleted {deleted_count} documents for user {user_id}",
            "deleted_count": deleted_count,
            "failed_deletions": failed_deletions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
