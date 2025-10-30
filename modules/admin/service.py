"""
Admin services for the Floor Plan Agent API
"""
import hashlib
import json
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import secrets
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status

from modules.config.settings import settings
from modules.database import db_manager, User
from modules.auth.service import auth_service
from modules.admin.models import UserStatus, FeedbackType, SubscriptionInterval

class AdminService:
    """Admin service class"""
    
    def __init__(self):
        self.db = db_manager
        self.jwt_secret = settings.JWT_SECRET_KEY
        self.token_expiry = timedelta(hours=24*7 )
    
    def admin_login(self, email: str, password: str) -> Dict[str, Any]:
        """Admin login process"""
        
        admin = self.db.verify_admin_credentials(email, password)
        if not admin:
            raise HTTPException(status_code=401, detail="Invalid admin credentials")
        
        # Update last login
        self.db.update_admin_last_login(admin.id)
        
        # Generate JWT token
        token = self.generate_admin_token(admin.id, admin.email, admin.is_super_admin)
        
        return {
            "message": "Admin login successful",
            "admin_id": admin.id,
            "username": admin.username,
            "email": admin.email,
            "is_super_admin": admin.is_super_admin,
            "access_token": token,
            "token": token,
            "token_type": "bearer"
        }
    
    def admin_register(self, username: str, email: str, password: str, confirm_password: str, is_super_admin: bool) -> Dict[str, Any]:
        """Admin registration process"""
        # Check if super admin exists (only super admin can create other admins)
        # super_admins = self.db.get_super_admins()
        # if not super_admins and not is_super_admin:
        #     raise HTTPException(status_code=403, detail="First admin must be a super admin")
        
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")
        
        if len(password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
        
        # Check if email already exists
        existing_admin = self.db.get_admin_by_email(email)
        if existing_admin:
            raise HTTPException(status_code=400, detail="Admin email already exists")
        
        try:
            admin_id = self.db.create_admin_user(username, email, password, is_super_admin)
            return {
                "message": "Admin created successfully",
                "admin_id": admin_id,
                "username": username,
                "email": email,
                "is_super_admin": is_super_admin
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    def generate_admin_token(self, admin_id: int, email: str, is_super_admin: bool) -> str:
        """Generate JWT token for admin"""
        payload = {
            "admin_id": admin_id,
            "email": email,
            "is_super_admin": is_super_admin,
            "exp": datetime.utcnow() + self.token_expiry
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_admin_token(self, token: str) -> bool:
        """Verify admin JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return bool(payload.get("admin_id"))
        except ExpiredSignatureError:
            return False
        except InvalidTokenError:
            return False
    
    def get_all_users(self, status: Optional[UserStatus] = None, search: Optional[str] = None) -> Dict[str, Any]:
        """Get all users with optional filtering and subscription details"""
        try:
            status_value = status.value if status else None
            users = self.db.get_all_users_with_filters(status_value, search)

            user_entries: List[Dict[str, Any]] = []

            for user in users:
                subscription = self.db.get_user_subscription(user.id)
                subscription_info = None

                if subscription:
                    plan = self.db.get_subscription_plan_by_id(subscription.plan_id)
                    plan_features = plan.features if plan else None
                    if isinstance(plan_features, str):
                        try:
                            plan_features = json.loads(plan_features)
                        except json.JSONDecodeError:
                            pass
                    plan_prices = [
                        {
                            "duration_months": price.duration_months,
                            "price": price.price,
                            "currency": price.currency,
                            "stripe_price_id": price.stripe_price_id,
                        }
                        for price in (plan.prices or [])
                    ] if plan else None

                    subscription_info = {
                        "id": subscription.id,
                        "status": subscription.status,
                        "interval": subscription.interval,
                        "auto_renew": subscription.auto_renew,
                        "current_period_start": subscription.current_period_start,
                        "current_period_end": subscription.current_period_end,
                        "plan": {
                            "id": plan.id if plan else None,
                            "name": plan.name if plan else None,
                            "storage_gb": plan.storage_gb if plan else None,
                            "project_limit": plan.project_limit if plan else None,
                            "action_limit": plan.action_limit if plan else None,
                            "features": plan_features,
                            "prices": plan_prices,
                        } if plan else None,
                    }

                user_entries.append({
                    "id": user.id,
                    "firstname": user.firstname,
                    "lastname": user.lastname,
                    "email": user.email,
                    "is_verified": user.is_verified,
                    "is_active": user.is_active,
                    "role": user.role,
                    "last_login": user.last_login,
                    "created_at": user.created_at,
                    "subscription": subscription_info,
                })

            return {
                "users": user_entries,
                "count": len(user_entries)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")
    
    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Delete a user and all their data"""
        try:
            # Get user data before deletion for AWS cleanup
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Delete user files from AWS (implement this in storage service)
            from modules.storage.service import storage_service
            from modules.projects.service import project_service

            storage_service.delete_user_files(user_id)

            project_cleanup: Optional[Dict[str, Any]] = None
            try:
                project_cleanup = project_service.delete_user_projects(user_id)
            except Exception as cleanup_error:
                # Continue with account deletion but surface cleanup issues to the caller
                project_cleanup = {
                    "message": "Failed to delete one or more projects",
                    "error": str(cleanup_error),
                }

            # Delete user from database
            success = self.db.delete_user(user_id)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to delete user")

            return {
                "message": "User deleted successfully",
                "user_id": user_id,
                "email": user.email,
                "project_cleanup": project_cleanup,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")
    
    def update_user_status(self, user_id: int, is_active: bool) -> Dict[str, Any]:
        """Update user active status"""
        try:
            success = self.db.update_user_status(user_id, is_active)
            if not success:
                raise HTTPException(status_code=404, detail="User not found")

            status = "active" if is_active else "inactive"
            return {
                "message": f"User status updated to {status}",
                "user_id": user_id,
                "is_active": is_active
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating user status: {str(e)}")

    def switch_user_subscription_plan(
        self,
        user_id: int,
        plan_id: int,
        interval: SubscriptionInterval = SubscriptionInterval.SIX_MONTH,
    ) -> Dict[str, Any]:
        """Switch a user's subscription plan without payment processing."""

        try:
            user = self.db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            plan = self.db.get_subscription_plan_by_id(plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail="Subscription plan not found")

            if isinstance(interval, SubscriptionInterval):
                normalized_interval = interval.value
            else:
                normalized_interval = SubscriptionInterval.normalize(interval).value

            now = datetime.utcnow()
            if normalized_interval == SubscriptionInterval.ANNUAL.value:
                period_end = now + timedelta(days=365)
            else:
                period_end = now + timedelta(days=182)

            existing_subscription = self.db.get_user_subscription(user_id)

            if existing_subscription:
                self.db.update_user_subscription(
                    existing_subscription.id,
                    plan_id=plan_id,
                    interval=normalized_interval,
                    current_period_start=now,
                    current_period_end=period_end,
                    status='active',
                    is_active=True,
                    auto_renew=existing_subscription.auto_renew,
                )
                subscription_id = existing_subscription.id
            else:
                subscription_id = self.db.create_user_subscription(
                    user_id,
                    plan_id,
                    interval=normalized_interval,
                    current_period_start=now,
                    current_period_end=period_end,
                    status='active',
                    auto_renew=True,
                    is_active=True,
                )

            # Ensure storage record exists for the user
            if not self.db.get_user_storage(user_id):
                self.db.create_user_storage(user_id)

            updated_subscription = self.db.get_user_subscription(user_id)

            self.db.log_user_activity(
                user_id,
                "subscription_updated_by_admin",
                {
                    "plan_id": plan_id,
                    "interval": normalized_interval,
                    "subscription_id": subscription_id,
                },
            )

            plan_features = plan.features
            if isinstance(plan_features, str):
                try:
                    plan_features = json.loads(plan_features)
                except json.JSONDecodeError:
                    pass

            return {
                "message": "User subscription updated successfully",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "interval": normalized_interval,
                "plan": {
                    "id": plan.id,
                    "name": plan.name,
                    "features": plan_features,
                    "prices": [
                        {
                            "duration_months": price.duration_months,
                            "price": price.price,
                            "currency": price.currency,
                            "stripe_price_id": price.stripe_price_id,
                        }
                        for price in (plan.prices or [])
                    ],
                },
                "subscription": updated_subscription,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating subscription: {str(e)}")

    def create_user_account(
        self,
        firstname: str,
        lastname: str,
        email: str,
        password: str,
        confirm_password: str,
        mark_verified: bool = False
    ) -> Dict[str, Any]:
        """Create a user account from the admin dashboard"""

        if not auth_service.validate_email_format(email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Passwords do not match")

        is_valid, error_msg = auth_service.validate_password_strength(password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        existing_user = self.db.get_user_by_email(email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        try:
            user_id = self.db.create_user(firstname, lastname, email, password)

            if mark_verified:
                self.db.verify_user_email(user_id)

            return {
                "message": "User created successfully",
                "user_id": user_id,
                "firstname": firstname,
                "lastname": lastname,
                "email": email,
                "verified": mark_verified
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")
    
    def get_user_storage_stats(self, user_id: int) -> Dict[str, Any]:
        """Get user storage usage statistics"""
        try:
            user_storage = self.db.get_user_storage(user_id)
            if not user_storage:
                raise HTTPException(status_code=404, detail="User storage record not found")

            user_subscription = self.db.get_user_subscription(user_id)
            if not user_subscription:
                raise HTTPException(status_code=404, detail="User has no active subscription")

            plan = self.db.get_subscription_plan_by_id(user_subscription.plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail="Subscription plan not found")

            storage_limit_mb = plan.storage_gb * 1024
            used_percentage = (
                (user_storage.used_storage_mb / storage_limit_mb) * 100
                if storage_limit_mb > 0 else 0.0
            )

            return {
                "user_id": user_id,
                "used_storage_mb": round(user_storage.used_storage_mb, 2),
                "storage_limit_mb": storage_limit_mb,
                "available_storage_mb": round(storage_limit_mb - user_storage.used_storage_mb, 2),
                "used_percentage": round(used_percentage, 2),
                "last_updated": user_storage.last_updated
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching storage stats: {str(e)}")

    def update_user_storage(self, user_id: int, file_size_mb: float) -> Dict[str, Any]:
        """Update user storage usage"""
        try:
            # Check if user has sufficient storage
            user_subscription = self.db.get_user_subscription(user_id)

            if not user_subscription:
                raise HTTPException(status_code=400, detail="User has no active subscription")

            plan = self.db.get_subscription_plan_by_id(user_subscription.plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail="Subscription plan not found")

            user_storage = self.db.get_user_storage(user_id)
            if not user_storage:
                self.db.create_user_storage(user_id)
                user_storage = self.db.get_user_storage(user_id)

            storage_limit_mb = plan.storage_gb * 1024
            new_usage = user_storage.used_storage_mb + file_size_mb

            if new_usage > storage_limit_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"Storage limit exceeded. Available: {storage_limit_mb - user_storage.used_storage_mb}MB, Required: {file_size_mb}MB"
                )

            # Update storage
            success = self.db.update_user_storage(user_id, new_usage)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update storage")

            return {
                "message": "Storage updated successfully",
                "user_id": user_id,
                "new_usage_mb": new_usage,
                "available_mb": storage_limit_mb - new_usage
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating storage: {str(e)}")
    
    def get_all_subscription_plans(self) -> Dict[str, Any]:
        """Get all subscription plans"""
        try:
            plans = self.db.get_all_subscription_plans()
            return {
                "plans": plans,
                "count": len(plans)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching subscription plans: {str(e)}")
    
    def create_subscription_plan(
        self,
        name: str,
        description: str,
        storage_gb: int,
        project_limit: int,
        user_limit: int,
        action_limit: int,
        features: List[str],
        has_free_trial: bool,
        trial_days: int,
        six_month_price: float,
        annual_price: float,
        six_month_stripe_price_id: Optional[str] = None,
        annual_stripe_price_id: Optional[str] = None,
        currency: str = "usd",
    ) -> Dict[str, Any]:
        """Create a new subscription plan"""

        try:
            prices = [
                {
                    "duration_months": SubscriptionInterval.SIX_MONTH.duration_months,
                    "price": six_month_price,
                    "currency": currency,
                    "stripe_price_id": six_month_stripe_price_id,
                },
                {
                    "duration_months": SubscriptionInterval.ANNUAL.duration_months,
                    "price": annual_price,
                    "currency": currency,
                    "stripe_price_id": annual_stripe_price_id,
                },
            ]

            plan_id = self.db.create_subscription_plan(
                name,
                description,
                storage_gb,
                project_limit,
                user_limit,
                action_limit,
                features,
                has_free_trial,
                trial_days,
                prices,
            )

            return {
                "message": "Subscription plan created successfully",
                "plan_id": plan_id,
                "name": name,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating subscription plan: {str(e)}")

    def update_subscription_plan(
        self,
        plan_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        storage_gb: Optional[int] = None,
        project_limit: Optional[int] = None,
        user_limit: Optional[int] = None,
        action_limit: Optional[int] = None,
        features: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        six_month_price: Optional[float] = None,
        annual_price: Optional[float] = None,
        six_month_stripe_price_id: Optional[str] = None,
        annual_stripe_price_id: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a subscription plan with partial data"""
        try:
            # Check if plan exists before update
            existing_plan = self.db.get_subscription_plan_by_id(plan_id)
            if not existing_plan:
                raise HTTPException(status_code=404, detail="Subscription plan not found")

            # Prepare update data from provided fields
            update_data = {}
            if name is not None:
                update_data['name'] = name
            if description is not None:
                update_data['description'] = description
            if storage_gb is not None:
                update_data['storage_gb'] = storage_gb
            if project_limit is not None:
                update_data['project_limit'] = project_limit
            if user_limit is not None:
                update_data['user_limit'] = user_limit
            if action_limit is not None:
                update_data['action_limit'] = action_limit
            if features is not None:
                update_data['features'] = features  # This will be JSON encoded in the DB manager
            if is_active is not None:
                update_data['is_active'] = is_active

            prices = []
            existing_prices = {price.duration_months: price for price in (existing_plan.prices or [])}

            if six_month_price is not None or six_month_stripe_price_id is not None:
                six_month_existing = existing_prices.get(SubscriptionInterval.SIX_MONTH.duration_months)
                prices.append({
                    "duration_months": SubscriptionInterval.SIX_MONTH.duration_months,
                    "price": six_month_price if six_month_price is not None else (six_month_existing.price if six_month_existing else 0.0),
                    "currency": (currency or (six_month_existing.currency if six_month_existing else "usd")),
                    "stripe_price_id": six_month_stripe_price_id if six_month_stripe_price_id is not None else (six_month_existing.stripe_price_id if six_month_existing else None),
                })

            if annual_price is not None or annual_stripe_price_id is not None:
                annual_existing = existing_prices.get(SubscriptionInterval.ANNUAL.duration_months)
                prices.append({
                    "duration_months": SubscriptionInterval.ANNUAL.duration_months,
                    "price": annual_price if annual_price is not None else (annual_existing.price if annual_existing else 0.0),
                    "currency": (currency or (annual_existing.currency if annual_existing else "usd")),
                    "stripe_price_id": annual_stripe_price_id if annual_stripe_price_id is not None else (annual_existing.stripe_price_id if annual_existing else None),
                })

            if prices:
                update_data['prices'] = prices
            elif currency is not None:
                # Apply currency change to existing prices
                update_data['prices'] = [
                    {
                        "duration_months": price.duration_months,
                        "price": price.price,
                        "currency": currency,
                        "stripe_price_id": price.stripe_price_id,
                    }
                    for price in existing_plan.prices or []
                ]

            # Perform the update
            success = self.db.update_subscription_plan(plan_id, **update_data)

            if not success:
                raise HTTPException(status_code=500, detail="Failed to update subscription plan")

            # Fetch and return the updated plan
            updated_plan = self.db.get_subscription_plan_by_id(plan_id)
            return {
                "message": "Subscription plan updated successfully",
                "plan_id": plan_id,
                "plan": updated_plan
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating subscription plan: {str(e)}")

    def delete_subscription_plan(self, plan_id: int) -> Dict[str, Any]:
        """Delete a subscription plan"""
        try:
            # First, check if the plan exists
            plan = self.db.get_subscription_plan_by_id(plan_id)
            if not plan:
                raise HTTPException(status_code=404, detail="Subscription plan not found")
            
            # Check if any users are currently subscribed to this plan
            # You might want to add this method to your DatabaseManager
            # active_subscriptions = self.db.get_active_subscriptions_count_by_plan(plan_id)
            # if active_subscriptions > 0:
            #     raise HTTPException(
            #         status_code=400, 
            #         detail=f"Cannot delete plan with {active_subscriptions} active subscriptions"
            #     )
            
            # Delete the plan from database
            success = self.db.delete_subscription_plan(plan_id)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to delete subscription plan")
            
            return {
                "message": "Subscription plan deleted successfully",
                "plan_id": plan_id,
                "plan_name": plan.name
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting subscription plan: {str(e)}")        
        
    def get_all_feedback(self, page: int = 1, limit: int = 20, rating: Optional[str] = None) -> Dict[str, Any]:
        """Get all feedback with pagination and filtering"""
        try:
            # Validate input parameters
            if page < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Page must be greater than 0"
                )
            
            if limit < 1 or limit > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Limit must be between 1 and 100"
                )
            
            # Validate rating if provided
            if rating and rating not in ['positive', 'negative']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Rating must be either 'positive' or 'negative'"
                )
            
            # Get feedback from database
            feedback_list = self.db.get_all_feedback(page, limit, rating)

            formatted_feedback = []
            user_cache: Dict[int, Optional[User]] = {}
            for feedback in feedback_list:
                if feedback.user_id in user_cache:
                    user = user_cache[feedback.user_id]
                else:
                    user = self.db.get_user_by_id(feedback.user_id)
                    user_cache[feedback.user_id] = user

                created_at = feedback.created_at
                if isinstance(created_at, datetime):
                    created_at_str = created_at.isoformat()
                elif created_at is None:
                    created_at_str = None
                else:
                    created_at_str = str(created_at)

                rating_value = feedback.rating.value if isinstance(feedback.rating, FeedbackType) else str(feedback.rating)

                formatted_feedback.append({
                    "id": feedback.id,
                    "user_id": feedback.user_id,
                    "rating": rating_value,
                    "comment": getattr(feedback, "comment", None) or feedback.project_name or "",
                    "ai_response": feedback.ai_response,
                    "created_at": created_at_str,
                    "user": {
                        "firstname": user.firstname if user else "",
                        "lastname": user.lastname if user else "",
                        "email": user.email if user else feedback.email
                    }
                })

            # Calculate pagination metadata
            total_feedback = self.db.get_total_feedback_count(rating)
            total_pages = (total_feedback + limit - 1) // limit if total_feedback > 0 else 1

            return {
                "feedback": formatted_feedback,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_feedback,
                    "pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1
                }
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log the error for debugging
            print(f"Error in get_all_feedback service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Unable to fetch feedback data"
            )
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            stats = self.db.get_feedback_statistics()
            total = stats.get('total', 0)
            positive = stats.get('positive', 0)
            negative = stats.get('negative', 0)
            
            positive_percentage = (positive / total * 100) if total > 0 else 0
            negative_percentage = (negative / total * 100) if total > 0 else 0
            
            return {
                "total_feedback": total,
                "positive": positive,
                "negative": negative,
                "positive_percentage": round(positive_percentage, 2),
                "negative_percentage": round(negative_percentage, 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching feedback statistics: {str(e)}")
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get admin dashboard statistics"""
        try:
            total_users = self.db.get_total_users_count()
            active_users = self.db.get_active_users_count()
            total_feedback = self.db.get_total_feedback_count()
            # recent_signups = self.db.get_recent_signups(7)  # Last 7 days
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "inactive_users": total_users - active_users,
                "total_feedback": total_feedback,
                # "recent_signups": recent_signups,
                "storage_usage": self.db.get_total_storage_usage()
            }
        except Exception as e:
            # raise HTTPException(status_code=500, detail=f"Error fetching dashboard statistics: {str(e)}")
                    # Log the actual error for debugging
            print(f"Database error in get_dashboard_statistics: {str(e)}")
            # Return generic error to client
            raise HTTPException(
                status_code=500,
                detail="Unable to fetch dashboard statistics"
            )

    def get_dashboard_metrics(self, days: int = 14) -> Dict[str, Any]:
        """Get time-series metrics for the admin dashboard graph."""

        try:
            return self.db.get_dashboard_metrics(days)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unable to fetch dashboard metrics: {str(e)}",
            )

    def get_subscription_reminders(self) -> Dict[str, Any]:
        """Get users who need subscription reminders"""
        try:
            # Users with subscriptions expiring in 30 days
            expiring_soon = self.db.get_subscriptions_expiring_soon(30)
            
            # Users with expired subscriptions (less than 21 days ago)
            recently_expired = self.db.get_recently_expired_subscriptions(21)
            
            return {
                "expiring_soon": expiring_soon,
                "recently_expired": recently_expired
            }
        except Exception as e:
            raise HTT
            


    # Add these methods to your AdminService class
    def get_ai_models(self) -> Dict[str, Any]:
        """Get all AI models"""
        try:
            models = self.db.get_all_ai_models()
            return {
                "models": models,
                "count": len(models)
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching AI models: {str(e)}"
            )

    def create_ai_model(self, name: str, provider: str, model_name: str, 
                    config: Dict[str, Any], is_active: bool = False) -> Dict[str, Any]:
        """Create a new AI model configuration"""
        try:
            # Validate that config is a dictionary
            if not isinstance(config, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Config must be a valid JSON object"
                )
            
            # If activating this model, deactivate others first
            if is_active:
                active_model = self.db.get_active_ai_model()
                if active_model:
                    self.db.activate_ai_model(0)  # Deactivate all first
            
            model_id = self.db.create_ai_model(name, provider, model_name, config, is_active)
            
            return {
                "message": "AI model created successfully",
                "model_id": model_id,
                "name": name,
                "provider": provider,
                "is_active": is_active
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating AI model: {str(e)}"
            )

    def activate_ai_model(self, model_id: int) -> Dict[str, Any]:
        """Activate an AI model"""
        try:
            # First check if model exists
            all_models = self.db.get_all_ai_models()
            model_exists = any(model.id == model_id for model in all_models)
            
            if not model_exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"AI model with ID {model_id} not found"
                )
            
            success = self.db.activate_ai_model(model_id)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to activate AI model"
                )
            
            return {
                "message": "AI model activated successfully",
                "model_id": model_id,
                "active": True
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error activating AI model: {str(e)}"
            )

    def get_active_ai_model(self) -> Dict[str, Any]:
        """Get the currently active AI model"""
        try:
            active_model = self.db.get_active_ai_model()
            
            if not active_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No active AI model found"
                )
            
            return {
                "active_model": active_model,
                "message": "Active AI model retrieved successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching active AI model: {str(e)}"
            )

    def delete_ai_model(self, model_id: int) -> Dict[str, Any]:
        """Delete an AI model configuration"""
        try:
            # First check if this is the active model
            active_model = self.db.get_active_ai_model()
            if active_model and active_model.id == model_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete the active AI model. Activate another model first."
                )
            
            success = self.db.delete_ai_model(model_id)
            
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"AI model with ID {model_id} not found"
                )
            
            return {
                "message": "AI model deleted successfully",
                "model_id": model_id,
                "deleted": True
            }
        except HTTPException:
            raise
        except Exception as e:
            if "Cannot delete the active AI model" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting AI model: {str(e)}"
            )
# Global admin service instance
admin_service = AdminService()

