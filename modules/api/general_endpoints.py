"""
General API endpoints for the Floor Plan Agent API
"""
import os
import io
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response, JSONResponse
from modules.config.settings import settings
from modules.config.utils import validate_file_path
from modules.pdf_processing.service import pdf_processor  # Import pdf_processor
from modules.database.models import db_manager
from modules.database.optimized_service import optimized_document_service
from modules.cache.document_cache import document_cache_manager

router = APIRouter(tags=["general"])


def _build_postman_collection(openapi_schema: dict, base_url: str) -> dict:
    """Convert the FastAPI OpenAPI schema into a Postman collection."""

    info = openapi_schema.get("info", {})
    collection = {
        "info": {
            "name": info.get("title", "Esticore API"),
            "description": info.get("description", "API documentation"),
            "version": info.get("version", "1.0.0"),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "item": [],
        "variable": [
            {"key": "baseUrl", "value": base_url}
        ],
    }

    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, details in path_item.items():
            if method.startswith("x-"):
                continue

            name = details.get("operationId") or details.get("summary") or f"{method.upper()} {path}"

            segments = []
            stripped = path.strip("/")
            if stripped:
                for segment in stripped.split("/"):
                    if segment.startswith("{") and segment.endswith("}"):
                        placeholder = segment[1:-1]
                        segments.append(f"{{{{{placeholder}}}}}")
                    else:
                        segments.append(segment)

            combined_parameters = []
            if isinstance(path_item, dict):
                combined_parameters.extend(path_item.get("parameters") or [])
            combined_parameters.extend(details.get("parameters") or [])

            query_params = []
            for parameter in combined_parameters:
                if parameter.get("in") == "query":
                    query_params.append(
                        {
                            "key": parameter.get("name"),
                            "value": f"{{{{{parameter.get('name')}}}}}",
                            "description": parameter.get("description"),
                            "disabled": True,
                        }
                    )

            request_entry = {
                "name": name,
                "request": {
                    "method": method.upper(),
                    "header": [],
                    "url": {
                        "raw": f"{{{{baseUrl}}}}{path}",
                        "host": ["{{baseUrl}}"],
                        "path": segments,
                        "query": query_params,
                    },
                    "description": details.get("description") or details.get("summary"),
                },
            }

            body_spec = details.get("requestBody")
            if isinstance(body_spec, dict):
                example = body_spec.get("example")

                if example is None:
                    content = body_spec.get("content")
                    if isinstance(content, dict):
                        for media_spec in content.values():
                            example = media_spec.get("example")
                            if example is None and isinstance(media_spec.get("examples"), dict):
                                first_example = next(iter(media_spec["examples"].values()), {})
                                example = first_example.get("value")
                            if example is not None:
                                break

                raw_body = ""
                if isinstance(example, (dict, list)):
                    raw_body = json.dumps(example, indent=2)
                elif isinstance(example, str):
                    raw_body = example

                request_entry["request"]["body"] = {
                    "mode": "raw",
                    "raw": raw_body,
                    "options": {"raw": {"language": "json"}},
                }

            collection["item"].append(request_entry)

    return collection

@router.get("/")
def root():
    """API root endpoint"""
    return {
        "message": f"{settings.APP_NAME} is running",
        "version": settings.VERSION,
        "status": "healthy"
    }

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION
    }


@router.get("/api/postman.json", response_class=JSONResponse)
async def postman_collection(request: Request):
    """Return a generated Postman collection for the API."""

    openapi_schema = request.app.openapi()
    base_url = str(request.base_url).rstrip("/")
    collection = _build_postman_collection(openapi_schema, base_url)
    return JSONResponse(content=collection)

@router.get("/download")
async def download_file(path: str):
    """
    Download files with security validation
    Only allows downloads from the DATA_DIR for security
    """
    print(f"DEBUG: Download request for path: {path}")
    print(f"DEBUG: Current DATA_DIR: {settings.DATA_DIR}")
    print(f"DEBUG: Path exists: {os.path.exists(path)}")
    
    # First, try the path as-is within DATA_DIR
    if validate_file_path(path, settings.DATA_DIR):
        filename = os.path.basename(path)
        print(f"DEBUG: Returning file using original path: {path}")
        return FileResponse(path, filename=filename)
    
    # If that fails, try common deployment path corrections
    # This handles cases where paths were stored with different working directories
    potential_paths = []
    
    # If the path looks like it's already absolute, try to find it relative to our dirs
    if os.path.isabs(path):
        filename_only = os.path.basename(path)
        potential_paths.extend([
            os.path.join(settings.DOCS_DIR, filename_only) if settings.DOCS_DIR else None,
            os.path.join(settings.OUTPUT_DIR, filename_only) if settings.OUTPUT_DIR else None,
            os.path.join(settings.IMAGES_DIR, filename_only),
        ])
        # Filter out None values
        potential_paths = [p for p in potential_paths if p is not None]
    
    # Try each potential path
    for potential_path in potential_paths:
        print(f"DEBUG: Trying potential path: {potential_path}")
        if validate_file_path(potential_path, settings.DATA_DIR):
            filename = os.path.basename(potential_path)
            print(f"DEBUG: Found file at corrected path: {potential_path}")
            return FileResponse(potential_path, filename=filename)
    
    print(f"DEBUG: File not found in any location")
    raise HTTPException(404, detail="File not found")

@router.post("/download/pdf")
async def download_pdf(user_id: int, doc_id: str):
    """
    Download a PDF document for a specific user and document ID
    """
    print(f"DEBUG: PDF download request for doc_id: {doc_id}, user_id: {user_id}")
    
    # Get document information
    try:
        document = db_manager.get_document_by_doc_id(doc_id)
        if not document:
            raise HTTPException(404, detail="Document not found")
        
        print(f"DEBUG: Document retrieved: {document.filename}")
        
        # Check if using database storage
        if pdf_processor.use_database_storage and document.file_id:
            # Serve from database
            try:
                file_content = pdf_processor.get_document_content(doc_id)
                
                # Create streaming response
                def generate():
                    yield file_content
                
                return StreamingResponse(
                    io.BytesIO(file_content),
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={document.filename}"}
                )
                
            except Exception as e:
                print(f"DEBUG: Error retrieving document from database: {e}")
                raise HTTPException(500, detail="Error retrieving document from database")
        
        else:
            # Legacy file storage - get document info for file paths
            doc_info = pdf_processor.get_document_info(doc_id)
            pdf_path = doc_info.get("pdf_path")
            
            if not pdf_path:
                raise HTTPException(404, detail="Document file path not found")
            
            print(f"DEBUG: Document pdf_path: {pdf_path}")
            print(f"DEBUG: File exists at pdf_path: {os.path.exists(pdf_path)}")
            
            # First, try the stored path as-is
            if validate_file_path(pdf_path, settings.DOCS_DIR):
                filename = os.path.basename(pdf_path)
                print(f"DEBUG: Returning PDF using stored path: {pdf_path}")
                return FileResponse(pdf_path, filename=filename)
            
            # If that fails, try to find the file by doc_id in our docs directory
            potential_pdf_paths = []
            if settings.DOCS_DIR:
                potential_pdf_paths.append(os.path.join(settings.DOCS_DIR, f"{doc_id}.pdf"))
            if settings.OUTPUT_DIR:
                potential_pdf_paths.append(os.path.join(settings.OUTPUT_DIR, f"{doc_id}.pdf"))
            
            # Also try looking for files that start with the doc_id (for annotation outputs)
            if settings.OUTPUT_DIR and os.path.exists(settings.OUTPUT_DIR):
                for filename in os.listdir(settings.OUTPUT_DIR):
                    if filename.startswith(doc_id) and filename.endswith('.pdf'):
                        potential_pdf_paths.append(os.path.join(settings.OUTPUT_DIR, filename))
            
            if settings.DOCS_DIR and os.path.exists(settings.DOCS_DIR):
                for filename in os.listdir(settings.DOCS_DIR):
                    if filename.startswith(doc_id) and filename.endswith('.pdf'):
                        potential_pdf_paths.append(os.path.join(settings.DOCS_DIR, filename))
            
            # Try each potential path
            for potential_path in potential_pdf_paths:
                print(f"DEBUG: Trying potential PDF path: {potential_path}")
                if os.path.exists(potential_path) and validate_file_path(potential_path, settings.DATA_DIR):
                    filename = os.path.basename(potential_path)
                    print(f"DEBUG: Found PDF at corrected path: {potential_path}")
                    return FileResponse(potential_path, filename=filename)
            
            print(f"DEBUG: PDF file not found in any location for doc_id: {doc_id}")
            raise HTTPException(404, detail="PDF file not found")
            
    except FileNotFoundError:
        print(f"DEBUG: Document {doc_id} not found in database")
        raise HTTPException(404, detail="Document not found")
    except Exception as e:
        print(f"DEBUG: Error in download_pdf: {e}")
        raise HTTPException(500, detail="Internal server error")


@router.get("/download/pdf")
async def download_pdf_optimized(user_id: int, doc_id: str, request: Request):
    """
    Optimized PDF download with caching and streaming (GET method)
    
    Args:
        user_id: User identifier for access control
        doc_id: Document identifier
        request: FastAPI request object for conditional requests
    
    Returns:
        StreamingResponse with PDF content and optimized headers
    """
    try:
        # Validate parameters
        if not doc_id or not user_id:
            raise HTTPException(400, detail="doc_id and user_id are required")
        
        # Check cache for metadata first
        metadata = await document_cache_manager.get_document_metadata(doc_id)
        
        if metadata:
            # Validate user access for owners and shared project members
            if not optimized_document_service.validate_user_access(doc_id, user_id):
                raise HTTPException(403, detail="Access denied")
        else:
            # Get metadata from database
            metadata = await optimized_document_service.get_document_metadata_only(doc_id, user_id)
            
            if not metadata:
                raise HTTPException(404, detail="Document not found")
            
            # Cache the metadata for future requests
            await document_cache_manager.cache_document_metadata(doc_id, metadata)
        
        # Check for conditional requests (ETag)
        if_none_match = request.headers.get("if-none-match")
        if if_none_match and if_none_match.strip('"') == metadata.etag:
            return Response(status_code=304)
        
        # Try to get content from cache
        content = await document_cache_manager.get_document_content(doc_id)
        
        if content:
            # Serve from cache
            response_headers = {
                "Content-Type": metadata.content_type,
                "Content-Disposition": f'attachment; filename="{metadata.filename}"',
                "Content-Length": str(len(content)),
                "Cache-Control": "private, max-age=3600",
                "ETag": f'"{metadata.etag}"',
                "Accept-Ranges": "bytes"
            }
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type=metadata.content_type,
                headers=response_headers
            )
        
        else:
            # Get content from database
            doc_with_content = await optimized_document_service.get_document_with_content(doc_id, user_id)
            
            if not doc_with_content:
                # Fallback to legacy file system approach
                return await _serve_from_filesystem(doc_id, user_id, metadata)
            
            content = doc_with_content.content
            
            # Cache content if it's small enough
            if doc_with_content.is_cacheable():
                await document_cache_manager.cache_document_content(doc_id, content)
            
            # Prepare response headers
            response_headers = {
                "Content-Type": metadata.content_type,
                "Content-Disposition": f'attachment; filename="{metadata.filename}"',
                "Content-Length": str(len(content)),
                "Cache-Control": "private, max-age=3600",
                "ETag": f'"{metadata.etag}"',
                "Accept-Ranges": "bytes"
            }
            
            return StreamingResponse(
                io.BytesIO(content),
                media_type=metadata.content_type,
                headers=response_headers
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(500, detail="Internal server error")


async def _serve_from_filesystem(doc_id: str, user_id: int, metadata):
    """
    Fallback method to serve PDF from filesystem (legacy storage)
    
    Args:
        doc_id: Document identifier
        user_id: User identifier
        metadata: Document metadata
    
    Returns:
        FileResponse for filesystem-based PDF
    """
    try:
        # Get document info for file paths (legacy approach)
        doc_info = pdf_processor.get_document_info(doc_id)
        pdf_path = doc_info.get("pdf_path")
        
        if not pdf_path:
            raise HTTPException(404, detail="Document file path not found")
        
        # Validate file path and existence
        if not validate_file_path(pdf_path, settings.DATA_DIR) or not os.path.exists(pdf_path):
            # Try alternative paths (same logic as original endpoint)
            potential_paths = []
            
            if settings.DOCS_DIR:
                potential_paths.append(os.path.join(settings.DOCS_DIR, f"{doc_id}.pdf"))
            if settings.OUTPUT_DIR:
                potential_paths.append(os.path.join(settings.OUTPUT_DIR, f"{doc_id}.pdf"))
            
            # Look for files that start with doc_id
            for directory in [settings.OUTPUT_DIR, settings.DOCS_DIR]:
                if directory and os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if filename.startswith(doc_id) and filename.endswith('.pdf'):
                            potential_paths.append(os.path.join(directory, filename))
            
            # Try each potential path
            pdf_path = None
            for potential_path in potential_paths:
                if os.path.exists(potential_path) and validate_file_path(potential_path, settings.DATA_DIR):
                    pdf_path = potential_path
                    break
            
            if not pdf_path:
                raise HTTPException(404, detail="PDF file not found")
        
        # Prepare response headers
        response_headers = {
            "Cache-Control": "private, max-age=3600",
            "ETag": f'"{metadata.etag}"',
            "Accept-Ranges": "bytes"
        }
        
        return FileResponse(
            pdf_path, 
            filename=metadata.filename,
            headers=response_headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail="Error serving file from filesystem")


@router.get("/download/output/{output_id}")
async def download_generated_output(output_id: str):
    """
    Download a generated output file by output ID
    """
    print(f"DEBUG: Generated output download request for output_id: {output_id}")
    
    if not pdf_processor.use_database_storage:
        raise HTTPException(501, detail="Generated output downloads only available with database storage")
    
    try:
        output_dict = pdf_processor.get_generated_output(output_id)
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(output_dict['file_data']),
            media_type=output_dict['content_type'],
            headers={"Content-Disposition": f"attachment; filename={output_dict['filename']}"}
        )
        
    except FileNotFoundError:
        print(f"DEBUG: Generated output {output_id} not found")
        raise HTTPException(404, detail="Generated output not found")
    except Exception as e:
        print(f"DEBUG: Error downloading generated output: {e}")
        raise HTTPException(500, detail="Error downloading generated output")

@router.get("/outputs/user/{user_id}")
async def list_user_generated_outputs(user_id: int):
    """
    List all generated outputs for a user
    """
    if not pdf_processor.use_database_storage:
        raise HTTPException(501, detail="Generated output listing only available with database storage")
    
    try:
        outputs = pdf_processor.list_user_generated_outputs(user_id)
        return {
            "user_id": user_id,
            "outputs": outputs,
            "count": len(outputs)
        }
    except Exception as e:
        print(f"DEBUG: Error listing user outputs: {e}")
        raise HTTPException(500, detail="Error listing generated outputs")
