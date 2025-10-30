"""Subscription services for the Floor Plan Agent API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import stripe
from fastapi import HTTPException

from modules.admin.models import SubscriptionInterval
from modules.config.settings import settings
from modules.database import db_manager

class SubscriptionService:
    """Subscription service class"""
    
    def __init__(self):
        self.db = db_manager
        self.stripe_api_key = settings.STRIPE_SECRET_KEY
        self.webhook_secret = settings.STRIPE_WEBHOOK_SECRET

        # Configure Stripe
        if self.stripe_api_key:
            stripe.api_key = self.stripe_api_key

    @staticmethod
    def _to_datetime(value: Any) -> Optional[datetime]:
        """Convert timestamps/strings to ``datetime`` objects."""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.utcfromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        """Normalize interval strings received from clients."""

        if not interval:
            raise HTTPException(status_code=400, detail="Billing interval is required")

        try:
            return SubscriptionInterval.normalize(interval).value
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @staticmethod
    def _from_stripe_interval(interval: Optional[str], interval_count: Optional[int] = None) -> Optional[str]:
        """Convert Stripe interval values to internal representations."""

        if interval is None:
            return None

        if interval == "month" and interval_count == 6:
            return SubscriptionInterval.SIX_MONTH.value

        if interval == "year" or (interval == "month" and interval_count == 12):
            return SubscriptionInterval.ANNUAL.value

        return None
    
    def get_available_plans(self) -> Dict[str, Any]:
        """Get all available subscription plans"""
        try:
            plans = self.db.get_all_subscription_plans()
            active_plans = [plan for plan in plans if plan.is_active]
            
            return {
                "plans": active_plans,
                "count": len(active_plans)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching plans: {str(e)}")
    
    def get_user_subscription(self, user_id: int) -> Dict[str, Any]:
        """Get user's current subscription"""
        try:
            subscription = self.db.get_user_subscription(user_id)
            user_storage = self.db.get_user_storage(user_id)

            if not subscription:
                # Check if user has a free trial
                user = self.db.get_user_by_id(user_id)
                created_at = self._to_datetime(getattr(user, "created_at", None)) if user else None

                if created_at and (datetime.now() - created_at).days <= 30:
                    # User is within trial period
                    trial_plan = self._get_trial_plan()
                    return {
                        "has_subscription": False,
                        "on_trial": True,
                        "trial_days_left": 30 - (datetime.now() - created_at).days,
                        "trial_plan": trial_plan,
                        "storage_usage": user_storage.used_storage_mb if user_storage else 0,
                    }

                return {
                    "has_subscription": False,
                    "on_trial": False,
                    "storage_usage": user_storage.used_storage_mb if user_storage else 0,
                }

            plan = self.db.get_subscription_plan_by_id(subscription.plan_id)
            period_end = self._to_datetime(subscription.current_period_end)

            days_until_expiry = (
                (period_end - datetime.now()).days
                if period_end and period_end > datetime.now()
                else 0
            )

            return {
                "has_subscription": True,
                "subscription": subscription,
                "plan": plan,
                "storage_usage": user_storage.used_storage_mb if user_storage else 0,
                "storage_limit_mb": plan.storage_gb * 1024 if plan else 0,
                "days_until_expiry": days_until_expiry,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching subscription: {str(e)}")

    def create_checkout_session(self, user_id: int, plan_id: int, interval: str) -> Dict[str, Any]:
        """Create Stripe checkout session"""
        if not self.stripe_api_key:
            raise HTTPException(status_code=500, detail="Stripe not configured")

        try:
            # Get user and plan details
            user = self.db.get_user_by_id(user_id)
            plan = self.db.get_subscription_plan_by_id(plan_id)

            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            if not plan:
                raise HTTPException(status_code=404, detail="Plan not found")

            normalized_interval = self._normalize_interval(interval)

            # Determine price based on interval
            plan_price = self.db.get_plan_price_option(plan_id, normalized_interval)
            if not plan_price:
                raise HTTPException(status_code=400, detail="Selected plan interval is not available")

            if plan_price.price <= 0 and not plan_price.stripe_price_id:
                raise HTTPException(status_code=400, detail="Plan pricing must be configured before checkout")

            if normalized_interval == SubscriptionInterval.ANNUAL.value:
                stripe_interval = "year"
                interval_count = 1
                interval_display = "annual"
            else:
                stripe_interval = "month"
                interval_count = 6
                interval_display = "6-month"

            currency = (plan_price.currency or "usd").lower()

            if plan_price.stripe_price_id:
                line_items = [{
                    'price': plan_price.stripe_price_id,
                    'quantity': 1,
                }]
            else:
                price_amount = int(plan_price.price * 100)
                line_items = [{
                    'price_data': {
                        'currency': currency,
                        'product_data': {
                            'name': plan.name,
                            'description': plan.description,
                        },
                        'unit_amount': price_amount,
                        'recurring': {
                            'interval': stripe_interval,
                            'interval_count': interval_count,
                        },
                    },
                    'quantity': 1,
                }]

            # Create Stripe checkout session
            checkout_session = stripe.checkout.Session.create(
                customer_email=user.email,
                payment_method_types=['card'],
                line_items=line_items,
                mode='subscription',
                success_url=f"{settings.FRONTEND_URL}/subscription/success?session_id={{CHECKOUT_SESSION_ID}}",
                cancel_url=f"{settings.FRONTEND_URL}/subscription/cancel",
                metadata={
                    'user_id': str(user_id),
                    'plan_id': str(plan_id),
                    'interval': normalized_interval,
                }
            )

            return {
                "checkout_url": checkout_session.url,
                "session_id": checkout_session.id,
                "message": f"Checkout session created for {plan.name} ({interval_display})",
            }
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating checkout session: {str(e)}")
    
    def handle_stripe_webhook(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events"""
        if not self.webhook_secret:
            raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")
        
        try:
            # Verify webhook signature
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            
            # Handle different event types
            if event['type'] == 'checkout.session.completed':
                return self._handle_checkout_session_completed(event)
            elif event['type'] == 'customer.subscription.updated':
                return self._handle_subscription_updated(event)
            elif event['type'] == 'customer.subscription.deleted':
                return self._handle_subscription_deleted(event)
            elif event['type'] == 'invoice.payment_succeeded':
                return self._handle_invoice_payment_succeeded(event)
            elif event['type'] == 'invoice.payment_failed':
                return self._handle_invoice_payment_failed(event)
            else:
                return {"status": "ignored", "event_type": event['type']}
                
        except stripe.error.SignatureVerificationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid signature: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Webhook error: {str(e)}")
    
    def _handle_checkout_session_completed(self, event) -> Dict[str, Any]:
        """Handle checkout.session.completed event"""
        session = event['data']['object']
        user_id = int(session['metadata']['user_id'])
        plan_id = int(session['metadata']['plan_id'])
        interval = session['metadata']['interval']
        
        # Create or update user subscription
        subscription = self.db.get_user_subscription(user_id)
        if subscription:
            # Update existing subscription
            self.db.update_user_subscription(
                subscription.id,
                plan_id=plan_id,
                stripe_subscription_id=session['subscription'],
                stripe_customer_id=session['customer'],
                interval=interval,
                status='active',
                is_active=True
            )
        else:
            # Create new subscription
            self.db.create_user_subscription(
                user_id, plan_id, session['subscription'], session['customer'], interval
            )
        
        # Ensure user has storage record
        if not self.db.get_user_storage(user_id):
            self.db.create_user_storage(user_id)

        return {"status": "success", "action": "subscription_created"}

    def _handle_subscription_updated(self, event) -> Dict[str, Any]:
        """Handle subscription updated event"""
        subscription = event['data']['object']

        # Update subscription in database
        stripe_subscription_id = subscription.get('id')
        if not stripe_subscription_id:
            return {"status": "ignored", "reason": "missing_subscription_id"}

        db_subscription = self.db.get_user_subscription_by_stripe_id(stripe_subscription_id)
        if not db_subscription:
            return {"status": "ignored", "reason": "subscription_not_found"}

        current_period_start = self._to_datetime(subscription.get('current_period_start'))
        current_period_end = self._to_datetime(subscription.get('current_period_end'))
        status = subscription.get('status', db_subscription.status)
        cancel_at_period_end = subscription.get('cancel_at_period_end', False)
        plan_info = (
            subscription.get('items', {})
            .get('data', [{}])[0]
            .get('plan', {})
        )
        interval = plan_info.get('interval', db_subscription.interval)
        interval_count = plan_info.get('interval_count')
        interval = self._from_stripe_interval(interval, interval_count) or db_subscription.interval

        self.db.update_user_subscription(
            db_subscription.id,
            stripe_customer_id=subscription.get('customer', db_subscription.stripe_customer_id),
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            status=status,
            interval=interval,
            auto_renew=not cancel_at_period_end,
            is_active=status in {'active', 'trialing'},
        )

        return {"status": "success", "action": "subscription_updated"}

    def _handle_subscription_deleted(self, event) -> Dict[str, Any]:
        """Handle subscription deletion events."""

        subscription = event['data']['object']
        stripe_subscription_id = subscription.get('id')
        if not stripe_subscription_id:
            return {"status": "ignored", "reason": "missing_subscription_id"}

        db_subscription = self.db.get_user_subscription_by_stripe_id(stripe_subscription_id)
        if not db_subscription:
            return {"status": "ignored", "reason": "subscription_not_found"}

        self.db.update_user_subscription(
            db_subscription.id,
            status='canceled',
            is_active=False,
            auto_renew=False,
        )

        return {"status": "success", "action": "subscription_deleted"}

    def _handle_invoice_payment_succeeded(self, event) -> Dict[str, Any]:
        """Handle successful invoice payments."""

        invoice = event['data']['object']
        stripe_subscription_id = invoice.get('subscription')
        if not stripe_subscription_id:
            return {"status": "ignored", "reason": "missing_subscription_id"}

        db_subscription = self.db.get_user_subscription_by_stripe_id(stripe_subscription_id)
        if not db_subscription:
            return {"status": "ignored", "reason": "subscription_not_found"}

        period_end = invoice.get('lines', {}).get('data', [{}])[0].get('period', {}).get('end')

        self.db.update_user_subscription(
            db_subscription.id,
            status='active',
            is_active=True,
            current_period_end=self._to_datetime(period_end) if period_end else db_subscription.current_period_end,
        )

        return {"status": "success", "action": "invoice_payment_succeeded"}

    def _handle_invoice_payment_failed(self, event) -> Dict[str, Any]:
        """Handle failed invoice payments."""

        invoice = event['data']['object']
        stripe_subscription_id = invoice.get('subscription')
        if not stripe_subscription_id:
            return {"status": "ignored", "reason": "missing_subscription_id"}

        db_subscription = self.db.get_user_subscription_by_stripe_id(stripe_subscription_id)
        if not db_subscription:
            return {"status": "ignored", "reason": "subscription_not_found"}

        self.db.update_user_subscription(
            db_subscription.id,
            status='past_due',
            is_active=False,
        )

        return {"status": "success", "action": "invoice_payment_failed"}

    def cancel_subscription(self, user_id: int) -> Dict[str, Any]:
        """Cancel the user's subscription at period end."""

        subscription = self.db.get_user_subscription(user_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        stripe_subscription_id = subscription.stripe_subscription_id

        if stripe_subscription_id and self.stripe_api_key:
            try:
                stripe.Subscription.modify(stripe_subscription_id, cancel_at_period_end=True)
            except stripe.error.StripeError as exc:
                raise HTTPException(status_code=400, detail=f"Stripe error: {str(exc)}")

        self.db.update_user_subscription(
            subscription.id,
            auto_renew=False,
            status=subscription.status or 'active',
        )

        return {
            "status": "success",
            "message": "Subscription will cancel at the end of the current period",
            "cancel_at_period_end": True,
        }

    def reactivate_subscription(self, user_id: int) -> Dict[str, Any]:
        """Reactivate a subscription that was set to cancel."""

        subscription = self.db.get_user_subscription(user_id)
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")

        stripe_subscription_id = subscription.stripe_subscription_id

        if stripe_subscription_id and self.stripe_api_key:
            try:
                stripe.Subscription.modify(stripe_subscription_id, cancel_at_period_end=False)
            except stripe.error.StripeError as exc:
                raise HTTPException(status_code=400, detail=f"Stripe error: {str(exc)}")

        self.db.update_user_subscription(
            subscription.id,
            auto_renew=True,
            status='active',
            is_active=True,
        )

        return {
            "status": "success",
            "message": "Subscription reactivated",
            "cancel_at_period_end": False,
        }

    def get_user_invoices(self, user_id: int, limit: int = 10) -> Dict[str, Any]:
        """Fetch invoices for the current user's Stripe customer."""

        if limit <= 0:
            raise HTTPException(status_code=400, detail="Limit must be greater than zero")

        subscription = self.db.get_user_subscription(user_id)
        if not subscription or not subscription.stripe_customer_id:
            raise HTTPException(status_code=404, detail="No Stripe customer found for user")

        if not self.stripe_api_key:
            raise HTTPException(status_code=500, detail="Stripe not configured")

        try:
            invoices = stripe.Invoice.list(customer=subscription.stripe_customer_id, limit=limit)
        except stripe.error.StripeError as exc:
            raise HTTPException(status_code=400, detail=f"Stripe error: {str(exc)}")

        formatted_invoices: List[Dict[str, Any]] = []
        invoice_items = (
            invoices.get('data', [])
            if isinstance(invoices, dict)
            else getattr(invoices, 'data', [])
        )
        for invoice in invoice_items:
            created_at = self._to_datetime(invoice.get('created'))
            formatted_invoices.append(
                {
                    "id": invoice.get('id'),
                    "status": invoice.get('status'),
                    "amount_paid": invoice.get('amount_paid', 0) / 100,
                    "currency": invoice.get('currency'),
                    "hosted_invoice_url": invoice.get('hosted_invoice_url'),
                    "created_at": created_at.isoformat() if created_at else None,
                }
            )

        return {
            "count": len(formatted_invoices),
            "invoices": formatted_invoices,
        }

    def _get_trial_plan(self) -> Optional[Any]:
        """Return the first plan marked as a free trial."""

        plans = self.db.get_all_subscription_plans()
        for plan in plans:
            if getattr(plan, 'has_free_trial', False):
                return plan
        return None

        

# Global subscription service instance
Subscription_Service = SubscriptionService()
