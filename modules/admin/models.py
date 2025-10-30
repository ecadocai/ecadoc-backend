"""
Admin data models for the Floor Plan Agent API
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class SubscriptionInterval(str, Enum):
    """Supported subscription billing intervals."""

    SIX_MONTH = "six_month"
    ANNUAL = "annual"

    @classmethod
    def normalize(cls, raw: str) -> "SubscriptionInterval":
        """Normalize arbitrary user-provided interval strings."""

        if not raw:
            raise ValueError("Interval is required")

        value = raw.strip().lower()
        mapping = {
            "6": cls.SIX_MONTH,
            "6m": cls.SIX_MONTH,
            "6-month": cls.SIX_MONTH,
            "six_month": cls.SIX_MONTH,
            "six-month": cls.SIX_MONTH,
            "semiannual": cls.SIX_MONTH,
            "semi-annual": cls.SIX_MONTH,
            "half-year": cls.SIX_MONTH,
            "half_year": cls.SIX_MONTH,
            "halfyear": cls.SIX_MONTH,
            "sixmonth": cls.SIX_MONTH,
            "biannual": cls.SIX_MONTH,
            "bi-annual": cls.SIX_MONTH,
            "annual": cls.ANNUAL,
            "1-year": cls.ANNUAL,
            "year": cls.ANNUAL,
            "yearly": cls.ANNUAL,
            "12": cls.ANNUAL,
            "12m": cls.ANNUAL,
            "12-month": cls.ANNUAL,
        }

        normalized = mapping.get(value)
        if not normalized:
            raise ValueError(f"Unsupported interval: {raw}")
        return normalized

    @property
    def duration_months(self) -> int:
        """Return the number of months for the interval."""

        return 6 if self is SubscriptionInterval.SIX_MONTH else 12


@dataclass
class SubscriptionPlanPrice:
    """Represents a priced billing option for a subscription plan."""

    id: Optional[int] = None
    plan_id: Optional[int] = None
    duration_months: int = 0
    price: float = 0.0
    currency: str = "usd"
    stripe_price_id: Optional[str] = None
    created_at: Optional[datetime] = None

class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

@dataclass
class AdminUser:
    """Admin user data model"""
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password: str = ""
    is_super_admin: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

@dataclass
class SubscriptionPlan:
    """Subscription plan data model"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    storage_gb: int = 0
    project_limit: int = 0
    user_limit: int = 1
    action_limit: int = 0
    features: List[str] = None
    is_active: bool = True
    has_free_trial: bool = False
    trial_days: int = 0
    created_at: Optional[datetime] = None
    prices: List[SubscriptionPlanPrice] = None

@dataclass
class UserSubscription:
    """User subscription data model"""
    id: Optional[int] = None
    user_id: int = 0
    plan_id: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_active: bool = True
    interval: SubscriptionInterval = SubscriptionInterval.SIX_MONTH
    auto_renew: bool = True
    created_at: Optional[datetime] = None

@dataclass
class UserStorage:
    """User storage usage data model"""
    id: Optional[int] = None
    user_id: int = 0
    used_storage_mb: int = 0
    last_updated: Optional[datetime] = None

@dataclass
class Feedback:
    """User feedback data model"""
    id: Optional[int] = None
    user_id: int = 0
    email: str = ""
    ai_response: str = ""
    rating: FeedbackType = FeedbackType.POSITIVE
    project_name: str = ""
    created_at: Optional[datetime] = None

@dataclass
class AIModel:
    """AI model configuration data model"""
    id: Optional[int] = None
    name: str = ""
    provider: str = ""
    model_name: str = ""
    is_active: bool = False
    config: Dict[str, Any] = None
    created_at: Optional[datetime] = None

@dataclass
class RecentlyViewedProject:
    """Recently viewed project tracking"""
    id: Optional[int] = None
    user_id: int = 0
    project_id: str = ""
    viewed_at: Optional[datetime] = None
    view_count: int = 1