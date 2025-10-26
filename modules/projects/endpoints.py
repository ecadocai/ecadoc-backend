"""
Project management API endpoints for the Floor Plan Agent API
"""

from typing import Optional, List
import os
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from modules.projects.service import project_service
from modules.agent.workflow import agent_workflow
from modules.database import db_manager
from modules.session import session_manager, context_resolver
from modules.auth.deps import get_current_user_id

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("/create")
async def create_project(
    project_name: str = Form(...),
    description: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    files: List[UploadFile] = File(default=[]),
):
    """
    Create a new project with optional multiple PDF uploads.
    """
    try:
        # Debug logs (optional)
        print(f"Debug: received files: {files}")
        print(f"Debug: files is None: {files is None}")
        print(f"Debug: files length: {len(files) if files else 0}")

        if files and len(files) > 0 and files[0].filename != "":
            # Validate file types and process files
            valid_files: List[UploadFile] = []
            for f in files:
                print(f"Debug: processing file: {f.filename}")
                if not f.filename.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail="Only PDF files are allowed")
                valid_files.append(f)

            print(f"Debug: valid files count: {len(valid_files)}")

            # Create project with PDFs
            result = project_service.create_project_with_pdfs(
                name=project_name,
                description=description,
                user_id=user_id,
                file_contents=[await f.read() for f in valid_files],
                filenames=[f.filename for f in valid_files],
            )
        else:
            print("Debug: No files provided, creating project without documents")
            # Create project without PDFs
            result = project_service.create_project_without_pdfs(
                name=project_name,
                description=description,
                user_id=user_id,
            )

        # Create a context-aware session for this project
        context_type, context_id = context_resolver.resolve_context({"project_id": result["project_id"]})
        session_id = session_manager.get_or_create_session(user_id, context_type, context_id)

        # Add the new, context-aware session_id to the response
        result["session_id"] = session_id

        # Side effects / logs
        db_manager.add_recently_viewed_project(user_id, result["project_id"])
        db_manager.log_user_activity(
            user_id,
            "project_created",
            {
                "project_id": result["project_id"],
                "project_name": project_name,
                "documents": len(result.get("documents") or []),
            },
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {str(e)}") from e


@router.post("/create-single")
async def create_project_single_file(
    project_name: str = Form(...),
    description: str = Form(...),
    user_id: int = Depends(get_current_user_id),
    file: UploadFile | None = File(None),
):
    """
    Create a new project with an optional single PDF upload (useful for testing).
    """
    try:
        print(f"Debug: received single file: {file}")
        print(f"Debug: file is None: {file is None}")
        if file:
            print(f"Debug: filename: {file.filename}")

        if file and file.filename and file.filename != "":
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")

            content = await file.read()
            print(f"Debug: file content size: {len(content)} bytes")

            result = project_service.create_project_with_pdfs(
                name=project_name,
                description=description,
                user_id=user_id,
                file_contents=[content],
                filenames=[file.filename],
            )
        else:
            print("Debug: No file provided, creating project without documents")
            result = project_service.create_project_without_pdfs(
                name=project_name,
                description=description,
                user_id=user_id,
            )

        # Create a context-aware session for this project
        context_type, context_id = context_resolver.resolve_context({"project_id": result["project_id"]})
        session_id = session_manager.get_or_create_session(user_id, context_type, context_id)

        # Add session_id to the response
        result["session_id"] = session_id

        # Side effects / logs
        db_manager.add_recently_viewed_project(user_id, result["project_id"])
        db_manager.log_user_activity(
            user_id,
            "project_created",
            {
                "project_id": result["project_id"],
                "project_name": project_name,
                "documents": 1 if result.get("documents") else 0,
            },
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project creation failed: {str(e)}") from e


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    user_id: int | None = None,
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Get project information by project ID.
    If `user_id` is provided and equals the authenticated user, try to reuse their session for this project.
    """
    try:
        project = project_service.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Return only an existing PROJECT-context session id (do NOT create one here).
        if user_id is not None:
            if user_id != current_user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied to requested user's project session",
                )
            session_id = session_manager.get_session_by_context(user_id, "PROJECT", project_id)
        else:
            session_id = session_manager.get_session_by_context(current_user_id, "PROJECT", project_id)

        # session_id may be None if no session was ever created for this project; return the project as-is in that case.
        project["session_id"] = session_id

        # Side effects / logs
        db_manager.add_recently_viewed_project(current_user_id, project_id)
        db_manager.log_user_activity(
            current_user_id,
            "project_viewed",
            {
                "project_id": project_id,
                "project_name": project.get("name"),
            },
        )

        return project

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/user/{user_id}")
async def get_user_projects(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    """
    Get all projects for a specific user.
    """
    try:
        if user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied to requested user's projects")
        projects = project_service.get_user_projects(user_id)
        return {"user_id": user_id, "projects": projects}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/shared")
async def get_shared_projects(user_id: int = Depends(get_current_user_id)):
    """
    Get projects shared with the authenticated user.
    """
    try:
        projects = project_service.get_shared_projects(user_id)
        return {"user_id": user_id, "projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{project_id}/upload-documents")
async def add_documents_to_project(
    project_id: str,
    user_id: int = Depends(get_current_user_id),
    files: List[UploadFile] = File(...),
):
    """
    Add one or more PDF documents to an existing project.
    """
    try:
        # Validate project access
        if not project_service.validate_project_access(project_id, user_id):
            raise HTTPException(status_code=403, detail="Access denied or project not found")

        # Validate file types and stream uploads to disk to avoid holding large files in memory
        valid_files: List[UploadFile] = []
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            valid_files.append(f)

        temp_paths: List[str] = []
        filenames: List[str] = []

        try:
            for f in valid_files:
                # Create a named temp file and stream the upload into it
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp_path = tmp.name
                tmp.close()
                # Stream in chunks to avoid reading entire file into memory
                with open(tmp_path, "wb") as out_f:
                    while True:
                        chunk = await f.read(64 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)

                temp_paths.append(tmp_path)
                filenames.append(f.filename)

            # Add documents to project using path-based upload
            result = project_service.add_documents_to_project_from_paths(
                project_id=project_id,
                file_paths=temp_paths,
                filenames=filenames,
                user_id=user_id,
            )
        finally:
            # Clean up any temp files that still exist (service may have moved them on success)
            for p in temp_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

        db_manager.log_user_activity(
            user_id,
            "project_documents_uploaded",
            {
                "project_id": project_id,
                "files": filenames,
            },
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}") from e


@router.post("/{project_id}/share")
async def share_project(
    project_id: str,
    inviter_id: int = Depends(get_current_user_id),
    invitee_email: str = Form(...),
    role: str = Form("member"),
):
    """
    Share a project with another user.
    """
    try:
        return project_service.share_project(project_id, inviter_id, invitee_email, role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to share project: {str(e)}") from e


@router.post("/invitations/{invitation_id}/respond")
async def respond_to_invitation(
    invitation_id: int,
    user_id: int = Depends(get_current_user_id),
    accept: bool = Form(...),
):
    """
    Accept or reject a project invitation.
    """
    try:
        return project_service.respond_to_invitation(invitation_id, user_id, accept)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to respond to invitation: {str(e)}") from e


@router.get("/invitations")
async def list_invitations(
    user_id: int,
    status: Optional[str] = None,
    current_user_id: int = Depends(get_current_user_id),
):
    """
    List invitations for a user.
    """
    try:
        if user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied to requested user's invitations")
        return project_service.list_user_invitations(user_id, status)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load invitations: {str(e)}") from e


@router.put("/{project_id}")
async def update_project(
    project_id: str,
    user_id: int = Depends(get_current_user_id),
    project_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Update project details (name and/or description).
    """
    try:
        # Validate project access
        if not project_service.validate_project_access(project_id, user_id):
            raise HTTPException(status_code=403, detail="Access denied or project not found")

        result = project_service.update_project(
            project_id=project_id,
            name=project_name,
            description=description,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project update failed: {str(e)}") from e


@router.get("/{project_id}/validate-access/{user_id}")
async def validate_project_access(
    project_id: str,
    user_id: int,
    current_user_id: int = Depends(get_current_user_id),
):
    """
    Check if a user has access to a project.
    """
    try:
        has_access = project_service.validate_project_access(project_id, user_id)
        return {
            "project_id": project_id,
            "user_id": user_id,
            "has_access": has_access,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    user_id: int = Depends(get_current_user_id),
    delete_shared_documents: bool = Form(True),
):
    """
    Delete a project and all its associated documents.

    Args:
        project_id: The project to delete
        user_id: User ID for ownership verification
        delete_shared_documents: Whether to delete documents that might be shared with other projects (default: True)
    """
    try:
        result = project_service.delete_project(project_id, user_id, delete_shared_documents)

        db_manager.log_user_activity(
            user_id,
            "project_deleted",
            {
                "project_id": project_id,
                "delete_shared_documents": delete_shared_documents,
            },
        )
        return result

    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg) from e
        if "access denied" in msg.lower():
            raise HTTPException(status_code=403, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/user/{user_id}")
async def delete_user_projects(user_id: int, current_user_id: int = Depends(get_current_user_id)):
    """
    Delete all projects for a specific user.
    """
    try:
        if user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Access denied to requested user's projects")
        result = project_service.delete_user_projects(user_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
