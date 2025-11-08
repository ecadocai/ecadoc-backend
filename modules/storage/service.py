"""
Storage services for the Floor Plan Agent API
"""
import os
from typing import Dict, Any, Optional
from fastapi import HTTPException, UploadFile
from datetime import datetime

from modules.config.settings import settings
from modules.database import db_manager
from modules.storage.aws_client import aws_client

class StorageService:
    """Storage service class"""
    
    def __init__(self):
        self.db = db_manager
        self.aws_client = aws_client
    
    def upload_file(self, user_id: int, file: UploadFile, project_id: str = None) -> Dict[str, Any]:
        """Upload file with storage validation"""
        try:
            # Check user storage limits
            user_storage = self.db.get_user_storage(user_id)
            user_subscription = self.db.get_user_subscription(user_id)
            
            if not user_storage:
                # Create storage record if it doesn't exist
                self.db.create_user_storage(user_id)
                user_storage = self.db.get_user_storage(user_id)
            
            # Get file size
            file.file.seek(0, 2)  # Seek to end
            file_size_mb = file.file.tell() / (1024 * 1024)  # Convert to MB
            file.file.seek(0)  # Reset to beginning
            
            # Check storage limits
            storage_limit_mb = self._get_user_storage_limit_mb(user_id, user_subscription)
            
            if user_storage.used_storage_mb + file_size_mb > storage_limit_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"Storage limit exceeded. Available: {storage_limit_mb - user_storage.used_storage_mb:.2f}MB, File: {file_size_mb:.2f}MB"
                )
            
            # Generate unique file name
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}{file_extension}"
            s3_key = f"user_{user_id}/{unique_filename}"
            
            # Upload to AWS S3 or local storage
            if self.aws_client.s3_client:
                # Upload to S3
                success = self.aws_client.upload_file(
                    file.file, 
                    s3_key, 
                    file.content_type
                )
                file_url = self.aws_client.get_file_url(s3_key)
            else:
                # Fallback to local storage
                success, file_url = self._save_file_locally(user_id, file, unique_filename)
            
            if not success:
                raise HTTPException(status_code=500, detail="File upload failed")
            
            # Update user storage
            new_usage = user_storage.used_storage_mb + file_size_mb
            self.db.update_user_storage(user_id, new_usage)

            self.db.log_user_activity(
                user_id,
                "file_uploaded",
                {
                    "filename": file.filename,
                    "file_size_mb": round(file_size_mb, 2),
                    "project_id": project_id,
                }
            )

            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "file_url": file_url,
                "file_size_mb": round(file_size_mb, 2),
                "storage_used_mb": round(new_usage, 2),
                "storage_available_mb": round(storage_limit_mb - new_usage, 2)
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")
    
    def delete_file(self, user_id: int, file_url: str) -> Dict[str, Any]:
        """Delete file and update storage"""
        try:
            # Extract file key from URL
            if "amazonaws.com" in file_url:
                # AWS S3 URL
                file_key = file_url.split("/")[-1]
                file_key = f"user_{user_id}/{file_key}"
                
                # Get file size before deletion
                file_size_mb = self.aws_client.get_file_size(file_key)
                if file_size_mb:
                    file_size_mb = file_size_mb / (1024 * 1024)  # Convert to MB
                else:
                    file_size_mb = 0
                
                # Delete from S3
                success = self.aws_client.delete_file(file_key)
            else:
                # Local file
                file_path = file_url.replace(settings.BASE_URL, "").lstrip("/")
                file_size_mb = self._get_local_file_size(file_path)
                success = self._delete_local_file(file_path)
            
            if not success:
                raise HTTPException(status_code=500, detail="File deletion failed")
            
            # Update user storage
            user_storage = self.db.get_user_storage(user_id)
            if user_storage and file_size_mb:
                new_usage = max(0, user_storage.used_storage_mb - file_size_mb)
                self.db.update_user_storage(user_id, new_usage)
            
            return {
                "message": "File deleted successfully",
                "file_size_freed_mb": round(file_size_mb, 2) if file_size_mb else 0,
                "storage_used_mb": round(new_usage, 2) if user_storage else 0
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)}")
    
    def get_user_storage_info(self, user_id: int) -> Dict[str, Any]:
        """Get user storage information"""
        try:
            user_storage = self.db.get_user_storage(user_id)
            user_subscription = self.db.get_user_subscription(user_id)
            
            if not user_storage:
                # Create storage record if it doesn't exist
                self.db.create_user_storage(user_id)
                user_storage = self.db.get_user_storage(user_id)
            
            storage_limit_mb = self._get_user_storage_limit_mb(user_id, user_subscription)
            used_percentage = (user_storage.used_storage_mb / storage_limit_mb) * 100 if storage_limit_mb > 0 else 0
            
            return {
                "user_id": user_id,
                "used_storage_mb": round(user_storage.used_storage_mb, 2),
                "storage_limit_mb": storage_limit_mb,
                "available_storage_mb": round(storage_limit_mb - user_storage.used_storage_mb, 2),
                "used_percentage": round(used_percentage, 2),
                "last_updated": user_storage.last_updated
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching storage info: {str(e)}")
    
    def delete_user_files(self, user_id: int) -> bool:
        """Delete all files for a user (when account is deleted)"""
        try:
            # Delete from AWS S3
            if self.aws_client.s3_client:
                self.aws_client.delete_user_files(user_id)
            
            # Delete local files
            user_dir = os.path.join(settings.UPLOAD_DIR, f"user_{user_id}")
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)
            
            # Reset storage usage
            self.db.update_user_storage(user_id, 0)
            
            return True
        except Exception as e:
            print(f"Error deleting user files: {str(e)}")
            return False
    
    def _get_user_storage_limit_mb(self, user_id: int, user_subscription) -> float:
        """Get user's storage limit in MB"""
        if user_subscription:
            plan = self.db.get_subscription_plan_by_id(user_subscription.plan_id)
            if plan:
                return plan.storage_gb * 1024  # Convert GB to MB
        
        # Default to free trial limits
        user = self.db.get_user_by_id(user_id)
        if user:
            created_at = user.created_at
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = None

            if created_at and (datetime.now() - created_at).days <= 30:
                return 15 * 1024  # 15GB trial

            if created_at is None:
                return 15 * 1024  # default trial when timestamp missing
        
        # No subscription - minimal storage
        return 100  # 100MB free tier
    
    def _save_file_locally(self, user_id: int, file: UploadFile, filename: str) -> tuple[bool, str]:
        """Save file to local storage"""
        try:
            user_dir = os.path.join(settings.UPLOAD_DIR, f"user_{user_id}")
            os.makedirs(user_dir, exist_ok=True)
            
            file_path = os.path.join(user_dir, filename)
            with open(file_path, "wb") as f:
                content = file.file.read()
                f.write(content)
            
            file_url = f"{settings.BASE_URL}/uploads/user_{user_id}/{filename}"
            return True, file_url
        except Exception as e:
            print(f"Local file save error: {str(e)}")
            return False, None
    
    def _get_local_file_size(self, file_path: str) -> float:
        """Get local file size in MB"""
        try:
            if os.path.exists(file_path):
                return os.path.getsize(file_path) / (1024 * 1024)
            return 0
        except:
            return 0
    
    def _delete_local_file(self, file_path: str) -> bool:
        """Delete local file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except:
            return False

# Global storage service instance
storage_service = StorageService()
