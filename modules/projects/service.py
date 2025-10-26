"""
Project management service for the Floor Plan Agent API
"""
import uuid
from typing import List, Optional, Dict, Any
from modules.database.models import db_manager, Project
from modules.pdf_processing.service import pdf_processor

class ProjectService:
    """Project management service"""
    
    def __init__(self):
        self.db = db_manager
    
    def create_project_with_pdf(self, name: str, description: str, user_id: int, 
                               file_content: bytes, filename: str) -> Dict[str, Any]:
        """Create a new project with an associated PDF document (backward compatibility)"""
        return self.create_project_with_pdfs(name, description, user_id, [file_content], [filename])
    
    def create_project_without_pdf(self, name: str, description: str, user_id: int) -> Dict[str, Any]:
        """Create a new project without an initial PDF document (backward compatibility)"""
        return self.create_project_without_pdfs(name, description, user_id)
    
    def create_project_with_pdfs(self, name: str, description: str, user_id: int, 
                               file_contents: List[bytes], filenames: List[str]) -> Dict[str, Any]:
        """Create a new project with one or more associated PDF documents"""
        # Generate unique project ID
        project_id = uuid.uuid4().hex
        
        print(f"Debug: create_project_with_pdfs called with {len(file_contents)} files")
        print(f"Debug: filenames: {filenames}")
        
        try:
            # First upload and index all PDFs
            doc_ids = []
            document_info = []
            
            for i, (file_content, filename) in enumerate(zip(file_contents, filenames)):
                print(f"Debug: processing file {i+1}/{len(file_contents)}: {filename}")
                print(f"Debug: file content size: {len(file_content)} bytes")
                
                pdf_result = pdf_processor.upload_and_index_pdf(file_content, filename, user_id)
                print(f"Debug: PDF processed successfully, doc_id: {pdf_result['doc_id']}")
                
                doc_ids.append(pdf_result["doc_id"])
                document_info.append({
                    "doc_id": pdf_result["doc_id"],
                    "filename": pdf_result["filename"],
                    "pages": pdf_result["pages"],
                    "chunks_indexed": pdf_result["chunks_indexed"]
                })
            
            print(f"Debug: All PDFs processed. doc_ids: {doc_ids}")
            
            # Create the project with the document IDs
            db_project_id = self.db.create_project(
                project_id=project_id,
                name=name,
                description=description,
                user_id=user_id,
                doc_ids=doc_ids  # Keep for backward compatibility
            )
            
            print(f"Debug: Project created in database with ID: {db_project_id}")
            
            if db_project_id is None:
                raise ValueError("Failed to create project in database")
            
            # Add documents to project using junction table
            for doc_id in doc_ids:
                self.db.add_document_to_project(project_id, doc_id)
            
            # Return comprehensive project information
            result = {
                "project_id": project_id,
                "name": name,
                "description": description,
                "user_id": user_id,
                "documents": document_info,
                "created_at": "just created"
            }
            
            print(f"Debug: Returning result: {result}")
            return result
            
        except Exception as e:
            print(f"Debug: Exception in create_project_with_pdfs: {e}")
            # If PDF upload failed or project creation failed, clean up
            # Note: This is a simplified cleanup that could be improved
            raise ValueError(f"Project creation failed: {str(e)}")
    
    def create_project_without_pdfs(self, name: str, description: str, user_id: int) -> Dict[str, Any]:
        """Create a new project without any initial PDF documents"""
        # Generate unique project ID
        project_id = uuid.uuid4().hex
        
        try:
            # Create the project without documents
            db_project_id = self.db.create_project(
                project_id=project_id,
                name=name,
                description=description,
                user_id=user_id,
                doc_ids=None
            )
            
            if db_project_id is None:
                raise ValueError("Failed to create project in database")
            
            # Return project information
            return {
                "project_id": project_id,
                "name": name,
                "description": description,
                "user_id": user_id,
                "documents": None,
                "created_at": "just created"
            }
            
        except Exception as e:
            raise ValueError(f"Project creation failed: {str(e)}")
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project information by project ID"""
        project = self.db.get_project_by_id(project_id)
        if not project:
            return None
        
        # Get document information from junction table
        documents = self.db.get_project_documents(project_id)
        document_info = []
        
        for doc in documents:
            doc_info = {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "pages": doc.pages,
                "status": doc.status
            }
            
            # Add storage-specific information
            if hasattr(doc, 'file_id') and doc.file_id:
                doc_info["file_id"] = doc.file_id
                doc_info["storage_type"] = "database"
            else:
                doc_info["storage_type"] = "filesystem"
            
            document_info.append(doc_info)
        
        # Include project members with basic user info (exclude password)
        members = self.db.get_project_members(project_id)
        member_info = []
        for member in members:
            try:
                user = self.db.get_user_by_id(member.user_id)
            except Exception:
                user = None

            member_entry = {
                "user_id": member.user_id,
                "role": member.role,
                "joined_at": member.created_at
            }

            if user:
                member_entry["user"] = {
                    "id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "profile_image": getattr(user, 'profile_image', None)
                }

            member_info.append(member_entry)

        # Include project owner/creator details (exclude password)
        owner = None
        try:
            owner_user = self.db.get_user_by_id(project.user_id)
            if owner_user:
                owner = {
                    "id": owner_user.id,
                    "firstname": owner_user.firstname,
                    "lastname": owner_user.lastname,
                    "profile_image": getattr(owner_user, 'profile_image', None)
                }
        except Exception:
            owner = None

        return {
            "project_id": project.project_id,
            "name": project.name,
            "description": project.description,
            "user_id": project.user_id,
            "owner": owner,
            "documents": document_info if document_info else None,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "members": member_info if member_info else None
        }
    
    def get_user_projects(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all projects for a user"""
        projects = self.db.get_user_projects(user_id)
        shared_projects = self.db.get_shared_projects_for_user(user_id)

        result = []
        for project in projects:
            # Get document information from junction table
            documents = self.db.get_project_documents(project.project_id)
            document_info = []

            for doc in documents:
                doc_info = {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "pages": doc.pages,
                    "status": doc.status,
                    "file_id": doc.file_id if hasattr(doc, 'file_id') else None,
                    "storage_type": "database" if hasattr(doc, 'file_id') and doc.file_id else "filesystem"
                }
                document_info.append(doc_info)
            
            # Attach owner details
            owner = None
            try:
                owner_user = self.db.get_user_by_id(project.user_id)
                if owner_user:
                    owner = {
                        "id": owner_user.id,
                        "firstname": owner_user.firstname,
                        "lastname": owner_user.lastname,
                        "profile_image": getattr(owner_user, 'profile_image', None),
                    }
            except Exception:
                owner = None

            result.append({
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "user_id": project.user_id,
                "owner": owner,
                "documents": document_info if document_info else None,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "access": "owner",
            })

        for shared in shared_projects:
            project = shared["project"]
            role = shared.get("role", "member")

            documents = self.db.get_project_documents(project.project_id)
            document_info = []

            for doc in documents:
                doc_info = {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "pages": doc.pages,
                    "chunks_indexed": doc.chunks_indexed,
                    "status": doc.status,
                    "file_id": doc.file_id if hasattr(doc, 'file_id') else None,
                    "storage_type": "database" if hasattr(doc, 'file_id') and doc.file_id else "filesystem"
                }
                document_info.append(doc_info)

            result.append({
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "user_id": project.user_id,
                "owner": None,
                "documents": document_info if document_info else None,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "access": "shared",
                "role": role
            })

        return result

    def get_shared_projects(self, user_id: int) -> List[Dict[str, Any]]:
        """Get projects shared with the specified user"""
        shared_projects = self.db.get_shared_projects_for_user(user_id)
        result: List[Dict[str, Any]] = []

        for shared in shared_projects:
            project = shared["project"]
            role = shared.get("role", "member")

            documents = self.db.get_project_documents(project.project_id)
            document_info = []

            for doc in documents:
                document_info.append({
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "pages": doc.pages,
                    "status": doc.status,
                    "file_id": getattr(doc, 'file_id', None),
                    "storage_type": "database" if getattr(doc, 'file_id', None) else "filesystem",
                })

            owner = None
            try:
                owner_user = self.db.get_user_by_id(project.user_id)
                if owner_user:
                    owner = {
                        "id": owner_user.id,
                        "firstname": owner_user.firstname,
                        "lastname": owner_user.lastname,
                        "profile_image": getattr(owner_user, 'profile_image', None),
                    }
            except Exception:
                owner = None

            result.append({
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "user_id": project.user_id,
                "owner": owner,
                "documents": document_info if document_info else None,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "access": "shared",
                "role": role,
            })

        return result
    
    def add_document_to_project(
        self,
        project_id: str,
        file_content: bytes,
        filename: str,
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add a PDF document to an existing project (backward compatibility)"""
        return self.add_documents_to_project(
            project_id,
            [file_content],
            [filename],
            user_id=user_id,
        )

    def add_documents_to_project(
        self,
        project_id: str,
        file_contents: List[bytes],
        filenames: List[str],
        user_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Add one or more PDF documents to an existing project"""
        # Check if project exists
        project = self.db.get_project_by_id(project_id)
        if not project:
            raise ValueError("Project not found")
        
        try:
            # Upload and index all PDFs
            document_info = []
            
            for i, (file_content, filename) in enumerate(zip(file_contents, filenames)):
                pdf_result = pdf_processor.upload_and_index_pdf(file_content, filename, project.user_id)
                
                document_info.append({
                    "doc_id": pdf_result["doc_id"],
                    "filename": pdf_result["filename"],
                    "pages": pdf_result["pages"],
                    "chunks_indexed": pdf_result["chunks_indexed"]
                })
            
            # Update the project with the new document IDs
            # Add documents to junction table
            for doc_info in document_info:
                self.db.add_document_to_project(project_id, doc_info["doc_id"])
            
            # Also update the projects table for backward compatibility
            doc_ids = project.doc_ids if project.doc_ids else []
            for doc_info in document_info:
                if doc_info["doc_id"] not in doc_ids:
                    doc_ids.append(doc_info["doc_id"])
            
            self.db.update_project_document(project_id, doc_ids)
            
            # Return updated project information
            updated_project = self.get_project(project_id)
            
            result = {
                "project_id": project_id,
                "documents": document_info
            }
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to add documents to project: {str(e)}")
    
    def update_project(self, project_id: str, name: str = None, description: str = None) -> Dict[str, Any]:
        """Update project details"""
        # Check if project exists
        project = self.db.get_project_by_id(project_id)
        if not project:
            raise ValueError("Project not found")
        
        try:
            # Update the project
            self.db.update_project_details(project_id, name, description)
            
            # Return updated project information
            return self.get_project(project_id)
            
        except Exception as e:
            raise ValueError(f"Failed to update project: {str(e)}")
    
    def validate_project_access(self, project_id: str, user_id: int) -> bool:
        """Check if a user has access to a project"""
        return self.db.user_has_project_access(project_id, user_id)
    
    def delete_project(self, project_id: str, user_id: int = None, delete_shared_documents: bool = True) -> Dict[str, Any]:
        """Delete a project and its associated documents

        Args:
            project_id: The project to delete
            user_id: User ID for ownership verification (optional)
            delete_shared_documents: Whether to delete documents that might be shared with other projects
        """
        # Check if project exists
        project = self.db.get_project_by_id(project_id)
        if not project:
            raise ValueError("Project not found")
        
        # Verify user access if user_id is provided
        if user_id is not None and project.user_id != user_id:
            raise ValueError("Access denied: You don't own this project")
        
        try:
            # Get all documents associated with this project
            documents = self.db.get_project_documents(project_id)
            
            deleted_documents = []
            failed_document_deletions = []
            skipped_shared_documents = []
            
            # Delete associated documents
            for doc in documents:
                try:
                    # Check if document is shared with other projects
                    doc_projects = self.db.get_document_projects(doc.doc_id)
                    is_shared = len(doc_projects) > 1
                    
                    if is_shared and not delete_shared_documents:
                        # Skip deletion of shared document, just remove from project
                        self.db.remove_document_from_project(project_id, doc.doc_id)
                        skipped_shared_documents.append({
                            "doc_id": doc.doc_id,
                            "filename": doc.filename,
                            "reason": "Document is shared with other projects"
                        })
                    else:
                        # Delete the document completely
                        success = pdf_processor.delete_document_files(doc.doc_id)
                        if success:
                            deleted_documents.append({
                                "doc_id": doc.doc_id,
                                "filename": doc.filename,
                                "was_shared": is_shared
                            })
                        else:
                            failed_document_deletions.append({
                                "doc_id": doc.doc_id,
                                "filename": doc.filename,
                                "error": "Failed to delete document files",
                                "was_shared": is_shared
                            })
                except Exception as e:
                    failed_document_deletions.append({
                        "doc_id": doc.doc_id,
                        "filename": doc.filename,
                        "error": str(e),
                        "was_shared": "unknown"
                    })
            
            # Delete project from database (this will cascade delete project_documents entries)
            project_deleted = self.db.delete_project(project_id)
            
            if not project_deleted:
                raise ValueError("Failed to delete project from database")
            
            return {
                "message": "Project deleted successfully",
                "project_id": project_id,
                "project_name": project.name,
                "deleted_documents": deleted_documents,
                "failed_document_deletions": failed_document_deletions,
                "skipped_shared_documents": skipped_shared_documents,
                "total_documents_processed": len(documents),
                "delete_shared_documents": delete_shared_documents
            }
            
        except Exception as e:
            raise ValueError(f"Failed to delete project: {str(e)}")

    def share_project(self, project_id: str, inviter_id: int, invitee_email: str, role: str = "member") -> Dict[str, Any]:
        """Invite another user to collaborate on a project"""
        project = self.db.get_project_by_id(project_id)
        if not project:
            raise ValueError("Project not found")

        if project.user_id != inviter_id:
            raise ValueError("Only the project owner can share this project")

        invitee = self.db.get_user_by_email(invitee_email)
        if not invitee:
            raise ValueError("Invitee email is not associated with an existing user")

        if invitee.id == inviter_id:
            raise ValueError("You cannot invite yourself to a project")

        if self.db.user_has_project_access(project_id, invitee.id):
            raise ValueError("User already has access to this project")

        invitation_id = self.db.create_project_invitation(project_id, inviter_id, invitee.id, invitee.email, role)

        metadata = {
            "project_id": project_id,
            "project_name": project.name,
            "invitation_id": invitation_id,
            "role": role,
            "inviter_id": inviter_id
        }

        inviter = self.db.get_user_by_id(inviter_id)
        if not inviter:
            raise ValueError("Inviter not found")
        title = f"Project invitation: {project.name}"
        message = f"{inviter.firstname} {inviter.lastname} invited you to collaborate on '{project.name}'."
        self.db.create_notification(invitee.id, title, message, "project_invitation", metadata)

        from modules.auth.email_service import email_service
        invitee_name = f"{invitee.firstname} {invitee.lastname}".strip()
        inviter_name = f"{inviter.firstname} {inviter.lastname}".strip()
        email_service.send_project_invitation_email(invitee.email, invitee_name or invitee.email, inviter_name or inviter.email, project.name)

        return {
            "message": "Invitation sent successfully",
            "invitation_id": invitation_id,
            "project_id": project_id,
            "invitee_id": invitee.id,
            "role": role,
            "status": "pending"
        }

    def respond_to_invitation(self, invitation_id: int, user_id: int, accept: bool) -> Dict[str, Any]:
        """Accept or reject a project invitation"""
        invitation = self.db.get_project_invitation_by_id(invitation_id)
        if not invitation:
            raise ValueError("Invitation not found")

        if invitation.invitee_id != user_id:
            raise ValueError("You are not authorized to respond to this invitation")

        if invitation.status != "pending":
            raise ValueError("Invitation has already been responded to")

        status = "accepted" if accept else "rejected"
        self.db.update_project_invitation_status(invitation_id, status)

        project = self.db.get_project_by_id(invitation.project_id)
        inviter = self.db.get_user_by_id(invitation.inviter_id)
        invitee = self.db.get_user_by_id(user_id)

        if not project:
            raise ValueError("Project not found")

        if accept:
            self.db.add_project_member(invitation.project_id, user_id, invitation.role)

        metadata = {
            "project_id": invitation.project_id,
            "project_name": project.name if project else None,
            "invitation_id": invitation_id,
            "invitee_id": user_id,
            "role": invitation.role,
            "status": status
        }

        title = f"Invitation {status}"
        invitee_name = f"{invitee.firstname} {invitee.lastname}".strip()
        message = f"{invitee_name or invitee.email} has {status} your invitation to '{project.name}'."
        self.db.create_notification(invitation.inviter_id, title, message, "project_invitation_update", metadata)

        return {
            "message": f"Invitation {status}",
            "project_id": invitation.project_id,
            "status": status,
            "role": invitation.role
        }

    def list_user_invitations(self, user_id: int, status: Optional[str] = None) -> Dict[str, Any]:
        """List invitations for a user"""
        invitations = self.db.get_user_project_invitations(user_id, status)
        formatted = []
        for invitation in invitations:
            project = self.db.get_project_by_id(invitation.project_id)
            inviter = self.db.get_user_by_id(invitation.inviter_id)
            formatted.append({
                "invitation_id": invitation.id,
                "project_id": invitation.project_id,
                "project_name": project.name if project else None,
                "inviter_id": invitation.inviter_id,
                "inviter_name": f"{inviter.firstname} {inviter.lastname}".strip() if inviter else None,
                "role": invitation.role,
                "status": invitation.status,
                "created_at": invitation.created_at,
                "responded_at": invitation.responded_at
            })

        return {"invitations": formatted}
    
    def delete_user_projects(self, user_id: int) -> Dict[str, Any]:
        """Delete all projects for a user"""
        try:
            projects = self.db.get_user_projects(user_id)
            
            deleted_projects = []
            failed_project_deletions = []
            total_documents_deleted = 0
            total_document_failures = 0
            
            for project in projects:
                try:
                    result = self.delete_project(project.project_id, user_id)
                    deleted_projects.append({
                        "project_id": project.project_id,
                        "project_name": project.name,
                        "documents_deleted": len(result["deleted_documents"]),
                        "document_failures": len(result["failed_document_deletions"])
                    })
                    total_documents_deleted += len(result["deleted_documents"])
                    total_document_failures += len(result["failed_document_deletions"])
                except Exception as e:
                    failed_project_deletions.append({
                        "project_id": project.project_id,
                        "project_name": project.name,
                        "error": str(e)
                    })
            
            return {
                "message": f"Processed {len(projects)} projects for user {user_id}",
                "user_id": user_id,
                "deleted_projects": deleted_projects,
                "failed_project_deletions": failed_project_deletions,
                "total_documents_deleted": total_documents_deleted,
                "total_document_failures": total_document_failures
            }
            
        except Exception as e:
            raise ValueError(f"Failed to delete user projects: {str(e)}")

# Global project service instance
project_service = ProjectService()