"""Subscription API endpoints for the Floor Plan Agent API."""

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from modules.admin.models import SubscriptionInterval
from modules.auth.deps import get_current_user_id
from modules.subscription.service import Subscription_Service

router = APIRouter(prefix="/subscription", tags=["subscription"])

@router.get("/plans")
async def get_subscription_plans():
    """Get all available subscription plans"""
    return Subscription_Service.get_available_plans()

@router.get("/user")
async def get_user_subscription(user_id: int = Depends(get_current_user_id)):
    """Get user's current subscription"""
    return Subscription_Service.get_user_subscription(user_id)

@router.post("/create-checkout-session")
async def create_checkout_session(
    plan_id: int = Form(...),
    interval: SubscriptionInterval = Form(SubscriptionInterval.SIX_MONTH),
    user_id: int = Depends(get_current_user_id)
):
    """Create Stripe checkout session"""
    interval_value = interval.value if isinstance(interval, SubscriptionInterval) else interval
    return Subscription_Service.create_checkout_session(user_id, plan_id, interval_value)

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        return Subscription_Service.handle_stripe_webhook(payload, sig_header)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cancel")
async def cancel_subscription(user_id: int = Depends(get_current_user_id)):
    """Cancel user subscription"""
    return Subscription_Service.cancel_subscription(user_id)

@router.post("/reactivate")
async def reactivate_subscription(user_id: int = Depends(get_current_user_id)):
    """Reactivate user subscription"""
    return Subscription_Service.reactivate_subscription(user_id)

@router.get("/invoices")
async def get_invoices(user_id: int = Depends(get_current_user_id), limit: int = 10):
    """Get user's payment invoices"""
    return Subscription_Service.get_user_invoices(user_id, limit)
