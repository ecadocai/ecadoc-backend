"""
Profile services for the Floor Plan Agent API
"""
import os
import secrets
from typing import Dict, Any
from fastapi import HTTPException, UploadFile
from datetime import datetime
from urllib.parse import urlparse

from modules.config.settings import settings
from modules.database import db_manager
from modules.storage.aws_client import aws_client

class ProfileService:
    """Profile service class"""
    
    def __init__(self):
        self.db = db_manager
        self.aws_client = aws_client
    
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get complete user profile"""
        try:
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            image_value = getattr(user, "profile_image", None)
            if image_value and not isinstance(image_value, str):
                image_value = str(image_value)

            if image_value:
                if image_value.startswith("http"):
                    profile_image_url = image_value
                else:
                    generated = self.aws_client.get_file_url(image_value)
                    profile_image_url = generated or image_value
            else:
                profile_image_url = None

            user_storage = self.db.get_user_storage(user_id)
            user_subscription = self.db.get_user_subscription(user_id)

            return {
                "user": {
                    "id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "email": user.email,
                    "is_verified": user.is_verified,
                    "profile_image": profile_image_url,
                    "created_at": user.created_at
                },
                "storage": {
                    "used_mb": user_storage.used_storage_mb if user_storage else 0,
                    "last_updated": user_storage.last_updated if user_storage else None
                } if user_storage else None,
                "subscription": user_subscription if user_subscription else None
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")
    
    def update_profile(self, user_id: int, firstname: str = None, lastname: str = None) -> Dict[str, Any]:
        """Update user profile"""
        try:
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Build update query
            update_data = {}
            if firstname is not None:
                update_data['firstname'] = firstname
            if lastname is not None:
                update_data['lastname'] = lastname
            
            if not update_data:
                raise HTTPException(status_code=400, detail="No data provided for update")
            
            # Update user in database
            success = self.db.update_user_profile(user_id, **update_data)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update profile")
            
            return {
                "message": "Profile updated successfully",
                "user_id": user_id,
                "updated_fields": list(update_data.keys())
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating profile: {str(e)}")
    
    def upload_profile_image(self, user_id: int, image: UploadFile) -> Dict[str, Any]:
        """Upload profile image"""
        try:
            # Validate image type
            allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
            if image.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Invalid image type. Allowed: JPEG, PNG, GIF, WebP")
            
            # Validate image size (max 5MB)
            max_size = 5 * 1024 * 1024  # 5MB
            image.file.seek(0, 2)
            file_size = image.file.tell()
            image.file.seek(0)
            
            if file_size > max_size:
                raise HTTPException(status_code=400, detail="Image too large. Maximum size: 5MB")
            
            # Generate unique filename
            file_extension = os.path.splitext(image.filename)[1]
            unique_filename = f"profile_{user_id}_{secrets.token_hex(8)}{file_extension}"
            s3_key = f"profile_images/{unique_filename}"
            
            # Upload image
            if self.aws_client.s3_client:
                success = self.aws_client.upload_file(
                    image.file, s3_key, image.content_type
                )
                image_url = self.aws_client.get_file_url(s3_key)
            else:
                success, image_url = self._save_image_locally(user_id, image, unique_filename)

            if not success:
                raise HTTPException(status_code=500, detail="Image upload failed")

            stored_value = s3_key if self.aws_client.s3_client and image_url else image_url
            self.db.update_user_profile(user_id, profile_image=stored_value)

            return {
                "message": "Profile image uploaded successfully",
                "image_url": image_url,
                "user_id": user_id
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")
    
    def delete_profile_image(self, user_id: int) -> Dict[str, Any]:
        """Delete profile image"""
        try:
            user = self.db.get_user_by_id(user_id)
            if not user or not user.profile_image:
                raise HTTPException(status_code=404, detail="No profile image found")
            
            stored_value = user.profile_image
            s3_key = None

            if stored_value:
                if not stored_value.startswith("http"):
                    s3_key = stored_value
                else:
                    parsed = urlparse(stored_value)
                    path = parsed.path.lstrip("/") if parsed.path else ""
                    if path:
                        s3_key = path

            if s3_key and self.aws_client.s3_client:
                self.aws_client.delete_file(s3_key)
            else:
                file_path = stored_value.replace(settings.BASE_URL or "", "").lstrip("/")
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Update user profile
            self.db.update_user_profile(user_id, profile_image=None)
            
            return {
                "message": "Profile image deleted successfully",
                "user_id": user_id
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")
    
    def get_recently_viewed_projects(self, user_id: int) -> Dict[str, Any]:
        """Get user's recently viewed projects"""
        try:
            recent_projects = self.db.get_recently_viewed_projects(user_id, limit=10)

            return {
                "recent_projects": recent_projects,
                "count": len(recent_projects)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching recent projects: {str(e)}")

    def get_user_activities(self, user_id: int, limit: int = 50) -> Dict[str, Any]:
        """Retrieve a user's recent activity log"""
        try:
            activities = self.db.get_user_activities(user_id, limit)

            formatted = []
            for activity in activities:
                formatted.append({
                    "id": activity.id,
                    "user_id": activity.user_id,
                    "action": activity.action,
                    "metadata": activity.metadata,
                    "created_at": activity.created_at,
                })

            return {
                "activities": formatted,
                "count": len(formatted)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching user activities: {str(e)}")
    
    def _save_image_locally(self, user_id: int, image: UploadFile, filename: str) -> tuple[bool, str]:
        """Save image to local storage"""
        try:
            profile_dir = os.path.join(settings.UPLOAD_DIR, "profile_images")
            os.makedirs(profile_dir, exist_ok=True)
            
            file_path = os.path.join(profile_dir, filename)
            with open(file_path, "wb") as f:
                content = image.file.read()
                f.write(content)
            
            image_url = f"{settings.BASE_URL}/uploads/profile_images/{filename}"
            return True, image_url
        except Exception as e:
            print(f"Local image save error: {str(e)}")
            return False, None

# Global profile service instance
profile_service = ProfileService()