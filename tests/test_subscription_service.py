import os
from types import SimpleNamespace

import pytest

from fastapi import HTTPException

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from modules.subscription.service import SubscriptionService
from modules.admin.models import SubscriptionInterval
from modules.database.models import SubscriptionPlanPrice


class DummyDB:
    def __init__(self, plan_price: SubscriptionPlanPrice, stripe_price: bool = True):
        self.plan_price = plan_price
        self.user = SimpleNamespace(id=1, email="user@example.com")
        self.plan = SimpleNamespace(name="Pro", description="Pro plan", storage_gb=10)
        self.calls = []

    def get_user_by_id(self, user_id: int):
        return self.user if user_id == 1 else None

    def get_subscription_plan_by_id(self, plan_id: int):
        return self.plan if plan_id == 2 else None

    def get_plan_price_option(self, plan_id: int, interval: str):
        self.calls.append((plan_id, interval))
        return self.plan_price

    def get_user_subscription(self, user_id: int):
        return None

    def create_user_subscription(self, *args, **kwargs):
        return 1

    def get_user_storage(self, user_id: int):
        return None

    def create_user_storage(self, user_id: int):
        return None


@pytest.fixture(autouse=True)
def patch_stripe(monkeypatch):
    class DummySession:
        def __init__(self, url: str):
            self.id = "sess_123"
            self.url = url

    captured = {}

    def fake_create(**kwargs):
        captured["kwargs"] = kwargs
        return DummySession("https://checkout.stripe.com/session")

    monkeypatch.setattr("stripe.checkout.Session.create", fake_create)
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: {})
    yield captured


def test_normalize_interval_variants():
    service = SubscriptionService()
    service.stripe_api_key = "sk_test"

    assert service._normalize_interval("six-month") == SubscriptionInterval.SIX_MONTH.value
    assert service._normalize_interval("ANNUAL") == SubscriptionInterval.ANNUAL.value
    with pytest.raises(HTTPException):
        service._normalize_interval("weekly")


def test_create_checkout_session_uses_stripe_price(patch_stripe):
    service = SubscriptionService()
    service.stripe_api_key = "sk_test"
    plan_price = SubscriptionPlanPrice(
        plan_id=2,
        duration_months=12,
        price=499.0,
        currency="usd",
        stripe_price_id="price_annual",
    )
    service.db = DummyDB(plan_price)

    result = service.create_checkout_session(user_id=1, plan_id=2, interval="annual")

    assert result["session_id"] == "sess_123"
    line_items = patch_stripe["kwargs"]["line_items"]
    assert line_items[0]["price"] == "price_annual"
    assert patch_stripe["kwargs"]["metadata"]["interval"] == SubscriptionInterval.ANNUAL.value


def test_create_checkout_session_builds_price_data_when_missing_stripe_id(patch_stripe):
    service = SubscriptionService()
    service.stripe_api_key = "sk_test"
    plan_price = SubscriptionPlanPrice(
        plan_id=2,
        duration_months=6,
        price=120.0,
        currency="usd",
        stripe_price_id=None,
    )
    service.db = DummyDB(plan_price)

    result = service.create_checkout_session(user_id=1, plan_id=2, interval="six_month")

    assert result["session_id"] == "sess_123"
    price_data = patch_stripe["kwargs"]["line_items"][0]["price_data"]
    assert price_data["unit_amount"] == 12000
    assert price_data["recurring"]["interval_count"] == 6
