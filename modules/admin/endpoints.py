"""
Admin API endpoints for the Floor Plan Agent API
"""
from fastapi import APIRouter, HTTPException, Depends, Form, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
import json

from modules.admin.service import admin_service
from modules.admin.models import UserStatus, FeedbackType, SubscriptionInterval

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to verify admin JWT token"""
    token = credentials.credentials
    if not admin_service.verify_admin_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired admin token")
    return token

@router.post("/login")
async def admin_login(email: str = Form(...), password: str = Form(...)):
    """Admin login endpoint"""
    print("route is hit")
    return admin_service.admin_login(email, password)

@router.post("/register")
async def admin_register(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    is_super_admin: bool = Form(False)
):
    """Admin registration endpoint (super admin only)"""
    return admin_service.admin_register(username, email, password, confirm_password, is_super_admin)

@router.get("/users", dependencies=[Depends(verify_admin_token)])
async def get_all_users(status: Optional[UserStatus] = None, search: Optional[str] = None):
    """Get all users with optional filtering"""
    return admin_service.get_all_users(status=status, search=search)

@router.post("/users", dependencies=[Depends(verify_admin_token)])
async def create_user(
    firstname: str = Form(...),
    lastname: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    mark_verified: bool = Form(False)
):
    """Admin endpoint to create a new user"""
    return admin_service.create_user_account(firstname, lastname, email, password, confirm_password, mark_verified)

@router.delete("/users/{user_id}", dependencies=[Depends(verify_admin_token)])
async def delete_user(user_id: int):
    """Delete a user and all their data"""
    return admin_service.delete_user(user_id)

@router.patch("/users/{user_id}/status", dependencies=[Depends(verify_admin_token)])
async def update_user_status(user_id: int, is_active: bool = Form(...)):
    """Update user active status"""
    return admin_service.update_user_status(user_id, is_active)

@router.get("/users/{user_id}/storage", dependencies=[Depends(verify_admin_token)])
async def get_user_storage(user_id: int):
    """Get user storage usage statistics"""
    return admin_service.get_user_storage_stats(user_id)

@router.post("/users/{user_id}/storage/update", dependencies=[Depends(verify_admin_token)])
async def update_user_storage(user_id: int, file_size_mb: float = Form(...)):
    """Update user storage usage (called when files are uploaded)"""
    return admin_service.update_user_storage(user_id, file_size_mb)

@router.post("/users/{user_id}/subscription/switch", dependencies=[Depends(verify_admin_token)])
async def switch_user_subscription(
    user_id: int,
    plan_id: int = Form(...),
    interval: SubscriptionInterval = Form(SubscriptionInterval.SIX_MONTH)
):
    """Switch a user's subscription plan without payment processing"""
    return admin_service.switch_user_subscription_plan(user_id, plan_id, interval)

@router.get("/subscription/plans", dependencies=[Depends(verify_admin_token)])
async def get_all_subscription_plans():
    """Get all subscription plans"""
    return admin_service.get_all_subscription_plans()

@router.post("/subscription/plans", dependencies=[Depends(verify_admin_token)])
async def create_subscription_plan(
    name: str = Form(...),
    description: str = Form(...),
    six_month_price: float = Form(...),
    annual_price: float = Form(...),
    storage_gb: int = Form(...),
    project_limit: int = Form(...),
    user_limit: int = Form(1),
    action_limit: int = Form(0),
    features: List[str] = Form([]),
    has_free_trial: bool = Form(False),
    trial_days: int = Form(0),
    six_month_stripe_price_id: Optional[str] = Form(None),
    annual_stripe_price_id: Optional[str] = Form(None),
    currency: str = Form("usd")
):
    """Create a new subscription plan"""
    
    print(features)

    return admin_service.create_subscription_plan(
        name,
        description,
        storage_gb,
        project_limit,
        user_limit,
        action_limit,
        features,
        has_free_trial,
        trial_days,
        six_month_price,
        annual_price,
        six_month_stripe_price_id,
        annual_stripe_price_id,
        currency,
    )
@router.patch("/subscription/plans/{plan_id}", dependencies=[Depends(verify_admin_token)])
async def update_subscription_plan(
    plan_id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    storage_gb: Optional[int] = Form(None),
    project_limit: Optional[int] = Form(None),
    user_limit: Optional[int] = Form(None),
    action_limit: Optional[int] = Form(None),
    features: Optional[List[str]] = Form(None),  # FastAPI can handle list in Form data
    is_active: Optional[bool] = Form(None),
    six_month_price: Optional[float] = Form(None),
    annual_price: Optional[float] = Form(None),
    six_month_stripe_price_id: Optional[str] = Form(None),
    annual_stripe_price_id: Optional[str] = Form(None),
    currency: Optional[str] = Form(None)
):
    """Update a subscription plan (partial update)"""
    # No need for manual JSON parsing for features list
    return admin_service.update_subscription_plan(
        plan_id,
        name,
        description,
        storage_gb,
        project_limit,
        user_limit,
        action_limit,
        features,
        is_active,
        six_month_price,
        annual_price,
        six_month_stripe_price_id,
        annual_stripe_price_id,
        currency,
    )
@router.delete("/subscription/plans/{plan_id}", dependencies=[Depends(verify_admin_token)])
async def delete_subscription_plan(plan_id: int):
    """Delete a subscription plan"""
    return admin_service.delete_subscription_plan(plan_id)

@router.get("/feedback", dependencies=[Depends(verify_admin_token)])
async def get_all_feedback(
    page: int = 1,
    limit: int = 20,
    rating: Optional[FeedbackType] = None
):
    """Get all feedback with pagination"""
    return admin_service.get_all_feedback(page, limit, rating)

@router.get("/feedback/stats", dependencies=[Depends(verify_admin_token)])
async def get_feedback_statistics():
    """Get feedback statistics (percentage of positive/negative)"""
    return admin_service.get_feedback_statistics()
@router.get("/ai/models", dependencies=[Depends(verify_admin_token)])
async def get_ai_models():
    """Get all AI models"""
    return admin_service.get_ai_models()

@router.get("/ai/models/active", dependencies=[Depends(verify_admin_token)])
async def get_active_ai_model():
    """Get the currently active AI model"""
    return admin_service.get_active_ai_model()

@router.post("/ai/models", dependencies=[Depends(verify_admin_token)])
async def create_ai_model(
    name: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    config: str = Form("{}"),
    is_active: bool = Form(False)
):
    """Create a new AI model configuration"""
    try:
        config_dict = json.loads(config) if config else {}
        return admin_service.create_ai_model(name, provider, model_name, config_dict, is_active)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in config field"
        )

@router.patch("/ai/models/{model_id}/activate", dependencies=[Depends(verify_admin_token)])
async def activate_ai_model(model_id: int):
    """Activate an AI model"""
    return admin_service.activate_ai_model(model_id)

@router.delete("/ai/models/{model_id}", dependencies=[Depends(verify_admin_token)])
async def delete_ai_model(model_id: int):
    """Delete an AI model configuration"""
    return admin_service.delete_ai_model(model_id)
    
@router.get("/dashboard/stats", dependencies=[Depends(verify_admin_token)])
async def get_dashboard_statistics():
    """Get admin dashboard statistics"""
    return admin_service.get_dashboard_statistics()


@router.get("/dashboard/metrics", dependencies=[Depends(verify_admin_token)])
async def get_dashboard_metrics(days: int = 14):
    """Get time-series metrics for admin dashboard charts."""
    return admin_service.get_dashboard_metrics(days)

@router.get("/subscription/reminders", dependencies=[Depends(verify_admin_token)])
async def get_subscription_reminders():
    """Get users who need subscription reminders"""
    return admin_service.get_subscription_reminders()

    