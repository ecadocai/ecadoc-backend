"""
Database models and operations for the Floor Plan Agent API
"""
import os
import sqlite3
import psycopg2
from psycopg2 import Error as PostgreSQLError
from psycopg2.extras import RealDictCursor
import hashlib
import json
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from modules.config.settings import settings
from modules.config.logger import logger
from psycopg2.pool import SimpleConnectionPool

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
        return 6 if self is SubscriptionInterval.SIX_MONTH else 12

class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

@dataclass
class User:
    """User data model"""
    id: Optional[int] = None
    firstname: str = ""
    lastname: str = ""
    email: str = ""
    password: str = ""
    google_id: Optional[str] = None
    is_verified: bool = False
    verification_token: Optional[str] = None
    verification_token_expires: Optional[datetime] = None
    created_at: Optional[datetime] = None
    is_active: bool = True
    profile_image: Optional[str] = None
    role: str = "user"
    last_login: Optional[datetime] = None

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
class SubscriptionPlanPrice:
    """Normalized price option for a subscription plan."""

    id: Optional[int] = None
    plan_id: Optional[int] = None
    duration_months: int = 0
    price: float = 0.0
    currency: str = "usd"
    stripe_price_id: Optional[str] = None
    created_at: Optional[datetime] = None

    @property
    def interval(self) -> SubscriptionInterval:
        return SubscriptionInterval.SIX_MONTH if self.duration_months == 6 else SubscriptionInterval.ANNUAL


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
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    status: str = "active"
    interval: str = SubscriptionInterval.SIX_MONTH.value
    auto_renew: bool = True
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class UserStorage:
    """User storage usage data model"""
    id: Optional[int] = None
    user_id: int = 0
    used_storage_mb: int = 0
    last_updated: Optional[datetime] = None

@dataclass
class UserActivity:
    """User activity log entry"""
    id: Optional[int] = None
    user_id: int = 0
    action: str = ""
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

@dataclass
class Feedback:
    """User feedback data model"""
    id: Optional[int] = None
    user_id: int = 0
    email: str = ""
    ai_response: str = ""
    rating: str = "positive"
    project_name: Optional[str] = None
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
    project_name: str = ""

@dataclass
class EmailVerificationToken:
    """Email verification token data model"""
    id: Optional[int] = None
    user_id: int = 0
    token: str = ""
    expires_at: datetime = None
    created_at: Optional[datetime] = None
    used_at: Optional[datetime] = None

@dataclass
class UserOTP:
    """Generic user OTP data model"""
    id: Optional[int] = None
    user_id: int = 0
    otp_code: str = ""
    purpose: str = ""
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None

@dataclass
class Document:
    """Document data model"""
    id: Optional[int] = None
    doc_id: str = ""  # Unique document identifier (UUID)
    filename: str = ""
    file_id: str = ""  # Reference to file_storage table
    pages: int = 0
    chunks_indexed: int = 0
    status: str = "active"  # active, file_missing, error
    user_id: int = 0  # Owner of the document
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ProjectMember:
    """Project membership data model"""
    id: Optional[int] = None
    project_id: str = ""
    user_id: int = 0
    role: str = "member"
    created_at: Optional[datetime] = None

@dataclass
class ProjectInvitation:
    """Project invitation data model"""
    id: Optional[int] = None
    project_id: str = ""
    inviter_id: int = 0
    invitee_id: Optional[int] = None
    invitee_email: str = ""
    role: str = "member"
    status: str = "pending"
    created_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None

@dataclass
class Notification:
    """User notification data model"""
    id: Optional[int] = None
    user_id: int = 0
    title: str = ""
    message: str = ""
    notification_type: str = "general"
    metadata: Optional[Dict[str, Any]] = None
    is_read: bool = False
    created_at: Optional[datetime] = None
    
    def validate(self) -> bool:
        """Validate notification data"""
        if self.user_id <= 0:
            return False
        if not isinstance(self.title, str) or not self.title.strip():
            return False
        if not isinstance(self.message, str) or not self.message.strip():
            return False
        # Accept any string type for extensibility
        if not isinstance(self.notification_type, str) or not self.notification_type.strip():
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'notification_type': self.notification_type,
            'metadata': self.metadata,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

@dataclass
class Project:
    """Project data model"""
    id: Optional[int] = None
    project_id: str = ""  # Unique project identifier
    name: str = ""
    description: str = ""
    user_id: int = 0
    doc_ids: Optional[List[str]] = None  # Associated document IDs as list
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

@dataclass
class ChatSession:
    """Enhanced chat session data model with context support"""
    id: Optional[int] = None
    session_id: str = ""
    user_id: int = 0
    context_type: str = ""  # PROJECT, DOCUMENT, GENERAL
    context_id: Optional[str] = None  # project_id, doc_id, or None for general
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None  # Additional context data

@dataclass
class ChatMessage:
    """Enhanced chat message data model with context support"""
    id: Optional[int] = None
    user_id: int = 0
    session_id: str = ""
    role: str = ""  # 'user' or 'assistant'
    message: str = ""
    timestamp: Optional[datetime] = None
    context_type: Optional[str] = None  # PROJECT, DOCUMENT, GENERAL
    context_id: Optional[str] = None  # project_id, doc_id, or None for general

@dataclass
class FileStorage:
    """File storage data model for binary file storage in database"""
    id: Optional[int] = None
    file_id: str = ""
    filename: str = ""
    content_type: str = ""
    file_size: int = 0
    file_data: bytes = b""
    user_id: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def validate(self) -> bool:
        """Validate file storage data"""
        if not self.file_id or not self.filename:
            return False
        if not self.content_type or not isinstance(self.file_data, bytes):
            return False
        if self.file_size != len(self.file_data):
            return False
        if self.user_id <= 0:
            return False
        return True
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding binary data"""
        result = {
            'id': self.id,
            'file_id': self.file_id,
            'filename': self.filename,
            'content_type': self.content_type,
            'file_size': self.file_size,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        if include_data:
            result['file_data'] = self.file_data
        return result

@dataclass
class VectorChunk:
    """Vector chunk data model for pgvector storage"""
    id: Optional[int] = None
    chunk_id: str = ""
    doc_id: str = ""
    page_number: int = 0
    chunk_text: str = ""
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    
    def validate(self) -> bool:
        """Validate vector chunk data"""
        if not self.chunk_id or not self.doc_id:
            return False
        if not self.chunk_text.strip():
            return False
        if self.page_number < 1:
            return False
        if self.embedding and len(self.embedding) != 1536:  # OpenAI embedding dimension
            return False
        return True
    
    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding embedding vector"""
        result = {
            'id': self.id,
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'page_number': self.page_number,
            'chunk_text': self.chunk_text,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_embedding and self.embedding:
            result['embedding'] = self.embedding
        return result

@dataclass
class GeneratedOutput:
    """Generated output data model for storing processed files"""
    id: Optional[int] = None
    output_id: str = ""
    filename: str = ""
    content_type: str = ""
    file_size: int = 0
    file_data: bytes = b""
    source_doc_id: Optional[str] = None
    user_id: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def validate(self) -> bool:
        """Validate generated output data"""
        if not self.output_id or not self.filename:
            return False
        if not self.content_type or not isinstance(self.file_data, bytes):
            return False
        if self.file_size != len(self.file_data):
            return False
        if self.user_id <= 0:
            return False
        return True
    
    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary, optionally excluding binary data"""
        result = {
            'id': self.id,
            'output_id': self.output_id,
            'filename': self.filename,
            'content_type': self.content_type,
            'file_size': self.file_size,
            'source_doc_id': self.source_doc_id,
            'user_id': self.user_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_data:
            result['file_data'] = self.file_data
        return result

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, db_name: str = None):
        self.db_name = db_name or settings.DATABASE_NAME
        self.use_rds = settings.USE_RDS

        # Determine database type based on port
        self.is_postgres = settings.DB_PORT == 5432
        self._pg_pool: SimpleConnectionPool | None = None
        
        if self.use_rds:
            if self.is_postgres:
                self.postgres_config = {
                    'host': settings.DB_HOST,
                    'port': settings.DB_PORT,
                    'database': settings.DB_NAME,
                    'user': settings.DB_USER,
                    'password': settings.DB_PASSWORD,
                    'connect_timeout': 30
                }
                # Initialize a PostgreSQL connection pool for better performance
                try:
                    minconn = 1
                    maxconn = max(settings.DB_POOL_SIZE, 5)
                    self._pg_pool = SimpleConnectionPool(
                        minconn=minconn,
                        maxconn=maxconn,
                        host=settings.DB_HOST,
                        port=settings.DB_PORT,
                        dbname=settings.DB_NAME,
                        user=settings.DB_USER,
                        password=settings.DB_PASSWORD,
                    )
                    logger.info("db_pool_initialized", extra={
                        "minconn": minconn,
                        "maxconn": maxconn,
                        "host": settings.DB_HOST,
                        "db": settings.DB_NAME,
                    })
                except Exception as e:
                    self._pg_pool = None
                    logger.error("db_pool_init_failed", extra={"error": str(e)})
        
        self.init_database()
        # One-time startup sequence sync for PostgreSQL to prevent PK collisions
        if self.use_rds and self.is_postgres:
            conn = None
            try:
                conn = self.get_connection()
                cur = conn.cursor()
                cur.execute(
                    "SELECT setval(pg_get_serial_sequence('userdata','id'), "
                    "COALESCE((SELECT MAX(id) FROM userdata), 1), "
                    "((SELECT MAX(id) FROM userdata) IS NOT NULL))"
                )
                conn.commit()
            except Exception as e:
                logger.warning("sequence_sync_failed", extra={"error": str(e)})
            finally:
                if conn:
                    self.release_connection(conn)

    @staticmethod
    def _map_user_row(row) -> Optional[User]:
        """Convert a database row tuple to a ``User`` instance."""
        if not row:
            return None

        role_index = 12 if len(row) > 12 else None
        last_login_index = 13 if len(row) > 13 else None

        role_value = row[role_index] if role_index is not None else None
        last_login_value = row[last_login_index] if last_login_index is not None else None

        profile_index = 11 if len(row) > 11 else None

        return User(
            id=row[0],
            firstname=row[1],
            lastname=row[2],
            email=row[3],
            password=row[4],
            google_id=row[5],
            is_verified=bool(row[6]),
            verification_token=row[7],
            verification_token_expires=row[8],
            created_at=row[9],
            is_active=bool(row[10]) if len(row) > 10 else True,
            profile_image=row[profile_index] if profile_index is not None else None,
            role=(role_value or "user"),
            last_login=last_login_value,
        )

    def get_connection(self):
        """Get database connection with retry logic and proper error handling"""
        if self.use_rds:
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    if self.is_postgres:
                        if self._pg_pool is not None:
                            # Try to obtain a pooled connection
                            try:
                                conn = self._pg_pool.getconn()
                            except Exception as pool_err:
                                # Pool may be temporarily exhausted; fall back to a direct connection
                                logger.warning(
                                    "db_pool_get_failed_fallback",
                                    extra={"error": str(pool_err)},
                                )
                                return psycopg2.connect(**self.postgres_config)

                            # Wrap close() to return to pool
                            try:
                                original_close = conn.close

                                def _release_back_to_pool(close_ref=original_close, pool=self._pg_pool, c=conn):
                                    try:
                                        pool.putconn(c)
                                    except Exception:
                                        # Fallback to actual close if pool put fails
                                        try:
                                            close_ref()
                                        except Exception:
                                            pass

                                setattr(conn, "_ecadoc_pooled", True)
                                conn.close = _release_back_to_pool  # type: ignore[assignment]
                            except Exception:
                                # If wrapping fails, return raw connection (still functional)
                                pass
                            return conn
                        return psycopg2.connect(**self.postgres_config)
                    else:
                        raise Exception("Only PostgreSQL is supported (configure DB_PORT=5432)")
                except (PostgreSQLError) as e:
                    if attempt == max_retries - 1:
                        db_type = "PostgreSQL"
                        raise Exception(f"Failed to connect to {db_type} after {max_retries} attempts: {e}")
                    
                    # Handle specific connection errors
                    import time
                    logger.warning(
                        "db_connect_retry",
                        extra={"attempt": attempt + 1, "retry_delay": retry_delay, "error": str(e)},
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                except Exception as e:
                    raise Exception(f"Unexpected database connection error: {e}")
        else:
            return sqlite3.connect(self.db_name)

    def release_connection(self, conn):
        """Release or close a database connection depending on backend/pool."""
        try:
            if getattr(conn, "_ecadoc_pooled", False) and self._pg_pool is not None:
                self._pg_pool.putconn(conn)
            else:
                conn.close()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    
    def execute_with_retry(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
        """Execute query with connection retry logic and proper error handling"""
        max_retries = 3
        
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self.get_connection()
                if self.is_postgres:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                else:
                    cur = conn.cursor()
                
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)
                
                if fetch_one:
                    result = cur.fetchone()
                elif fetch_all:
                    result = cur.fetchall()
                else:
                    result = cur.rowcount
                
                conn.commit()
                return result
                
            except (PostgreSQLError) as e:
                if conn:
                    conn.rollback()
                
                # Handle specific database errors that might be retryable
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))  # Progressive delay
                    continue
                else:
                    db_type = "PostgreSQL"
                    raise Exception(f"{db_type} query error: {e}")
            except Exception as e:
                if conn:
                    conn.rollback()
                raise Exception(f"Database query error: {e}")
            finally:
                if conn:
                    self.release_connection(conn)

    # -------- Jobs helpers --------
    def create_job(self, job_id: str, user_id: int, status: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            if self.use_rds and self.is_postgres:
                cur.execute(
                    "INSERT INTO jobs (id, user_id, status, error, metadata) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (job_id, user_id, status, None, json.dumps(metadata or {})),
                )
            else:
                # SQLite/MySQL: simple insert, ignore if exists
                try:
                    cur.execute(
                        f"INSERT INTO jobs (id, user_id, status, error, metadata) VALUES ({self._get_placeholder()}, {self._get_placeholder()}, {self._get_placeholder()}, NULL, {self._get_placeholder()})",
                        (job_id, user_id, status, json.dumps(metadata or {})),
                    )
                except Exception:
                    pass
            conn.commit()
            return True
        finally:
            self.release_connection(conn)

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            if self.use_rds and self.is_postgres:
                cur.execute(
                    "UPDATE jobs SET status=%s, error=%s, metadata=COALESCE(metadata, '{}'::jsonb) || %s::jsonb, updated_at=NOW() WHERE id=%s",
                    (status, error, json.dumps(metadata or {}), job_id),
                )
            else:
                # SQLite/MySQL
                cur.execute(
                    f"UPDATE jobs SET status={self._get_placeholder()}, error={self._get_placeholder()}, metadata={self._get_placeholder()}, updated_at=CURRENT_TIMESTAMP WHERE id={self._get_placeholder()}",
                    (status, error, json.dumps(metadata or {}), job_id),
                )
            conn.commit()
            return cur.rowcount > 0
        finally:
            self.release_connection(conn)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT id, user_id, status, error, metadata, created_at, updated_at FROM jobs WHERE id = {self._get_placeholder()}",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            # Row shape differs by backend; attempt to normalize
            if isinstance(row, dict):
                meta = row.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = None
                return {
                    "id": row.get("id"),
                    "user_id": row.get("user_id"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                    "metadata": meta,
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            else:
                # Positional tuple result
                meta = row[4]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = None
                return {
                    "id": row[0],
                    "user_id": row[1],
                    "status": row[2],
                    "error": row[3],
                    "metadata": meta,
                    "created_at": row[5],
                    "updated_at": row[6],
                }
        finally:
            self.release_connection(conn)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cur = conn.cursor()

        if self.use_rds and self.is_postgres:
            # PostgreSQL table creation statements
            logger.info("db_init", extra={"engine": "postgresql"})
            
            # Ensure pgvector extension is available
            self.ensure_pgvector_extension()
            
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS userdata(
                    id SERIAL PRIMARY KEY,
                    firstname VARCHAR(255) NOT NULL,
                    lastname VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    google_id VARCHAR(255) UNIQUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verification_token VARCHAR(255),
                    verification_token_expires TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    profile_image TEXT,
                    role VARCHAR(50) DEFAULT 'user',
                    last_login TIMESTAMP NULL
                )
            """)
            
            # Create admin_users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    is_super_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL
                )
            """)

            # Create subscription_plans table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plans (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    storage_gb INTEGER DEFAULT 0,
                    project_limit INTEGER DEFAULT 0,
                    user_limit INTEGER DEFAULT 1,
                    action_limit INTEGER DEFAULT 0,
                    features JSONB,
                    is_active BOOLEAN DEFAULT TRUE,
                    has_free_trial BOOLEAN DEFAULT FALSE,
                    trial_days INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create subscription_plan_prices table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                    id SERIAL PRIMARY KEY,
                    plan_id INTEGER NOT NULL,
                    duration_months INTEGER NOT NULL,
                    price NUMERIC(10, 2) NOT NULL,
                    currency VARCHAR(10) DEFAULT 'usd',
                    stripe_price_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (plan_id, duration_months),
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_prices_plan ON subscription_plan_prices (plan_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_prices_duration ON subscription_plan_prices (duration_months)")

            interval_column = self._quote_identifier("interval")
            # Create user_subscriptions table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    plan_id INTEGER NOT NULL,
                    stripe_subscription_id VARCHAR(255),
                    stripe_customer_id VARCHAR(255),
                    current_period_start TIMESTAMP,
                    current_period_end TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active',
                    {interval_column} VARCHAR(20) DEFAULT 'six_month',
                    auto_renew BOOLEAN DEFAULT TRUE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_subscriptions_user ON user_subscriptions (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_subscriptions_plan ON user_subscriptions (plan_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_subscriptions_status ON user_subscriptions (status)")

            # Create chat history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chathistory(
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_type VARCHAR(20) CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL')),
                    context_id VARCHAR(255),
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for chathistory
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chathistory_context ON chathistory (context_type, context_id)")
            
            # Create projects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects(
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(255) UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    user_id INTEGER NOT NULL,
                    doc_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)
            
            # Create documents table (updated schema)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents(
                    id SERIAL PRIMARY KEY,
                    doc_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_id VARCHAR(255),
                    pages INTEGER DEFAULT 0,
                    chunks_indexed INTEGER DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'active',
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for documents
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)")
            
            # Create project_documents junction table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_documents(
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
                    UNIQUE (project_id, doc_id)
                )
            """)
            
            # Create indexes for project_documents
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_documents_project_id ON project_documents (project_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_documents_doc_id ON project_documents (doc_id)")

            # Jobs table for async task status (MySQL)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id INT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    error TEXT,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_jobs_user (user_id),
                    INDEX idx_jobs_status (status)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Jobs table for async task status (SQLite)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Jobs table for async task status
            cur.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    error TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)")

            # Create project_members table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_members(
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    user_id INTEGER NOT NULL,
                    role VARCHAR(50) DEFAULT 'member',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    UNIQUE (project_id, user_id)
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members (user_id)")

            # Create project_invitations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_invitations(
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    inviter_id INTEGER NOT NULL,
                    invitee_id INTEGER,
                    invitee_email VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'member',
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    responded_at TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (inviter_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (invitee_id) REFERENCES userdata (id) ON DELETE SET NULL
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_invitations_invitee ON project_invitations (invitee_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_invitations_email ON project_invitations (invitee_email)")

            # Create notifications table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications(
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    notification_type VARCHAR(50) DEFAULT 'general',
                    metadata JSONB,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications (user_id, is_read)")

            # Create user OTP table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_otp_codes(
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    otp_code VARCHAR(10) NOT NULL,
                    purpose VARCHAR(50) NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    consumed_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_otp_lookup ON user_otp_codes (user_id, purpose, consumed_at)")

            # Create user_storage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_storage (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER UNIQUE NOT NULL,
                    used_storage_mb INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_storage_user ON user_storage (user_id)")

            # Create user_activity table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    action VARCHAR(255) NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_created ON user_activity (created_at)")

            # Create chat_sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions(
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    context_type VARCHAR(20) NOT NULL CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL')),
                    context_id VARCHAR(255),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for chat_sessions
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_context ON chat_sessions (user_id, context_type, context_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions (session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions (last_activity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_active ON chat_sessions (user_id, is_active)")
            
            conn.commit()
            
            # Create new tables for file and vector storage
            self.create_new_tables()
            
            # Migrate documents table if needed
            self.migrate_documents_table()
            
           
                
        elif self.use_rds:
            # MySQL table creation statements
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS userdata(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    firstname VARCHAR(255) NOT NULL,
                    lastname VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    google_id VARCHAR(255) UNIQUE,
                    is_verified BOOLEAN DEFAULT FALSE,
                    verification_token VARCHAR(255),
                    verification_token_expires TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    profile_image TEXT,
                    role VARCHAR(50) DEFAULT 'user',
                    last_login TIMESTAMP NULL
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create admin_users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    is_super_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL,
                    INDEX idx_email (email),
                    INDEX idx_super_admin (is_super_admin)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create subscription_plans table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plans (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    storage_gb INT DEFAULT 0,
                    project_limit INT DEFAULT 0,
                    user_limit INT DEFAULT 1,
                    action_limit INT DEFAULT 0,
                    features JSON,
                    is_active BOOLEAN DEFAULT TRUE,
                    has_free_trial BOOLEAN DEFAULT FALSE,
                    trial_days INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_name (name),
                    INDEX idx_is_active (is_active)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create subscription_plan_prices table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plan_id INT NOT NULL,
                    duration_months INT NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    currency VARCHAR(10) DEFAULT 'usd',
                    stripe_price_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_plan_duration (plan_id, duration_months),
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create user_subscriptions table - FIXED: interval is a reserved word, using backticks
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    plan_id INT NOT NULL,
                    stripe_subscription_id VARCHAR(255),
                    stripe_customer_id VARCHAR(255),
                    current_period_start TIMESTAMP,
                    current_period_end TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'active',
                    `interval` VARCHAR(20) DEFAULT 'six_month',
                    auto_renew BOOLEAN DEFAULT TRUE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id),
                    INDEX idx_user_id (user_id),
                    INDEX idx_plan_id (plan_id),
                    INDEX idx_stripe_subscription (stripe_subscription_id),
                    INDEX idx_status (status)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create user_storage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_storage (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT UNIQUE NOT NULL,
                    used_storage_mb INT DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create user_activity table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    action VARCHAR(255) NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_activity_user (user_id),
                    INDEX idx_activity_created (created_at)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    ai_response TEXT NOT NULL,
                    rating ENUM('positive', 'negative') NOT NULL,
                    project_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_user_id (user_id),
                    INDEX idx_rating (rating),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create ai_models table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    provider VARCHAR(100) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT FALSE,
                    config JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_provider (provider),
                    INDEX idx_is_active (is_active)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create recently_viewed_projects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recently_viewed_projects (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    project_id VARCHAR(255) NOT NULL,
                    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    view_count INT DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_user_project (user_id, project_id),
                    INDEX idx_user_id (user_id),
                    INDEX idx_viewed_at (viewed_at)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create chat history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chathistory(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    session_id TEXT NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_type ENUM('PROJECT', 'DOCUMENT', 'GENERAL') NULL,
                    context_id VARCHAR(255) NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_context (context_type, context_id)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create projects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS projects(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    project_id VARCHAR(255) UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    user_id INT NOT NULL,
                    doc_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    doc_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    pdf_path TEXT NOT NULL,
                    vector_path TEXT,
                    pages INT DEFAULT 0,
                    chunks_indexed INT DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'active',
                    user_id INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_doc_id (doc_id),
                    INDEX idx_user_id (user_id),
                    INDEX idx_status (status)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create project_documents junction table for many-to-many relationship
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_documents(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_project_document (project_id, doc_id),
                    INDEX idx_project_id (project_id),
                    INDEX idx_doc_id (doc_id)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create project_members table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_members(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    user_id INT NOT NULL,
                    role VARCHAR(50) DEFAULT 'member',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    UNIQUE KEY unique_member (project_id, user_id),
                    INDEX idx_project_members_user (user_id)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create project_invitations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_invitations(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    inviter_id INT NOT NULL,
                    invitee_id INT NULL,
                    invitee_email VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'member',
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    responded_at TIMESTAMP NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (inviter_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (invitee_id) REFERENCES userdata (id) ON DELETE SET NULL,
                    INDEX idx_project_invitations_invitee (invitee_id),
                    INDEX idx_project_invitations_email (invitee_email)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create notifications table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    notification_type VARCHAR(50) DEFAULT 'general',
                    metadata JSON,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_notifications_user (user_id, is_read)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create user OTP table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_otp_codes(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    otp_code VARCHAR(10) NOT NULL,
                    purpose VARCHAR(50) NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    consumed_at TIMESTAMP NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    INDEX idx_user_otp_lookup (user_id, purpose, consumed_at)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create chat_sessions table for enhanced session management
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id INT NOT NULL,
                    context_type ENUM('PROJECT', 'DOCUMENT', 'GENERAL') NOT NULL,
                    context_id VARCHAR(255) NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    metadata JSON NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE,
                    INDEX idx_user_context (user_id, context_type, context_id),
                    INDEX idx_session_id (session_id),
                    INDEX idx_last_activity (last_activity),
                    INDEX idx_active_sessions (user_id, is_active)
                ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            # Create test user if not exists
            cur.execute("SELECT * FROM userdata WHERE email = %s", ("test@example.com",))
            if not cur.fetchone():
                test_password = hashlib.sha256("testuser1".encode()).hexdigest()
                cur.execute(
                    "INSERT INTO userdata (firstname, lastname, email, password) VALUES (%s, %s, %s, %s)",
                    ("Test", "User", "test@example.com", test_password)
                )
            
            # Create default admin user if not exists
            cur.execute("SELECT * FROM admin_users WHERE email = %s", ("admin@esticore.com",))
            if not cur.fetchone():
                admin_password = hashlib.sha256("admin123".encode()).hexdigest()
                cur.execute(
                    "INSERT INTO admin_users (username, email, password, is_super_admin) VALUES (%s, %s, %s, %s)",
                    ("admin", "admin@esticore.com", admin_password, True)
                )
            
            # Create default AI models
            default_models = [
                ("GPT-4", "OpenAI", "gpt-4", True, '{"temperature": 0.7, "max_tokens": 2000}'),
                ("GPT-3.5-Turbo", "OpenAI", "gpt-3.5-turbo", False, '{"temperature": 0.7, "max_tokens": 2000}'),
                ("Claude-2", "Anthropic", "claude-2", False, '{"temperature": 0.7, "max_tokens": 2000}')
            ]
            
            for name, provider, model_name, is_active, config in default_models:
                cur.execute("SELECT * FROM ai_models WHERE name = %s", (name,))
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO ai_models (name, provider, model_name, is_active, config) VALUES (%s, %s, %s, %s, %s)",
                        (name, provider, model_name, is_active, config)
                    )
            
            # Create default subscription plans
            default_plans = [
                {
                    "name": "Solo Plan",
                    "description": "Project Repository & Document Upload (up to 3 active projects, 15GB storage)",
                    "six_month_price": 0.0,
                    "annual_price": 99.0,
                    "storage_gb": 15,
                    "project_limit": 3,
                    "user_limit": 1,
                    "action_limit": 350,
                    "features": [
                        "Version Control for document updates",
                        "Natural Language AI Chat (multi-turn chat, source citations)",
                        "Embedded PDF Viewer",
                        "Manual annotation Tools",
                        "Limited AI-Driven Markup/measurements (350 action limit per month)",
                        "Standard Email Support",
                        "Full Bluebeam Studio Sync",
                    ],
                    "has_free_trial": True,
                    "trial_days": 30,
                },
                {
                    "name": "Team Plan",
                    "description": "Team Collaboration (shared project folders, concurrent editing)",
                    "six_month_price": 98.0,
                    "annual_price": 499.0,
                    "storage_gb": 50,
                    "project_limit": 10,
                    "user_limit": 5,
                    "action_limit": 2000,
                    "features": [
                        "Everything in Solo Plan",
                        "Team Collaboration (shared project folders, concurrent editing)",
                        "Expanded Storage (50GB)",
                        "Increased AI-Driven mark-ups/Measurement(2000 action limit per month)",
                        "Bluebeam Studio Export (push AI-marked PDFs)",
                        "Priority Email + Chat Support",
                        "Project History Retention: up to 10 versions per project",
                    ],
                    "has_free_trial": False,
                    "trial_days": 0,
                },
            ]

            for plan in default_plans:
                cur.execute("SELECT id FROM subscription_plans WHERE name = %s", (plan["name"],))
                exists = cur.fetchone()
                if not exists:
                    plan_id = self.create_subscription_plan(
                        plan["name"],
                        plan["description"],
                        plan["storage_gb"],
                        plan["project_limit"],
                        plan["user_limit"],
                        plan["action_limit"],
                        plan["features"],
                        plan["has_free_trial"],
                        plan["trial_days"],
                        [
                            {"duration_months": 6, "price": plan["six_month_price"], "currency": "usd"},
                            {"duration_months": 12, "price": plan["annual_price"], "currency": "usd"},
                        ],
                    )
                    print(f"✓ Created default subscription plan {plan['name']} (id={plan_id})")
        else:
            # SQLite table creation statements
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS userdata(
                    id INTEGER PRIMARY KEY,
                    firstname VARCHAR(255) NOT NULL,
                    lastname VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    google_id VARCHAR(255) UNIQUE,
                    is_verified BOOLEAN DEFAULT 0,
                    verification_token VARCHAR(255),
                    verification_token_expires DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    profile_image TEXT,
                    role TEXT DEFAULT 'user',
                    last_login DATETIME
                )
            """)
            
            # Create admin_users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS admin_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_super_admin BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME
                )
            """)
            
            # Create subscription_plans table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    storage_gb INTEGER DEFAULT 0,
                    project_limit INTEGER DEFAULT 0,
                    user_limit INTEGER DEFAULT 1,
                    action_limit INTEGER DEFAULT 0,
                    features TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    has_free_trial BOOLEAN DEFAULT 0,
                    trial_days INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create subscription_plan_prices table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER NOT NULL,
                    duration_months INTEGER NOT NULL,
                    price REAL NOT NULL,
                    currency TEXT DEFAULT 'usd',
                    stripe_price_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(plan_id, duration_months),
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                )
            """)

            # Create user_subscriptions table - FIXED: interval is a reserved word, using backticks
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    plan_id INTEGER NOT NULL,
                    stripe_subscription_id TEXT,
                    stripe_customer_id TEXT,
                    current_period_start DATETIME,
                    current_period_end DATETIME,
                    status TEXT DEFAULT 'active',
                    `interval` TEXT DEFAULT 'six_month',
                    auto_renew BOOLEAN DEFAULT 1,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (plan_id) REFERENCES subscription_plans (id)
                )
            """)
            
            # Create user_storage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_storage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL,
                    used_storage_mb INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            # Create user_activity table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_created ON user_activity (created_at)")

            # Create feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    email TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    rating TEXT CHECK(rating IN ('positive', 'negative')) NOT NULL,
                    project_name TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)
            
            # Create ai_models table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 0,
                    config TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create recently_viewed_projects table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recently_viewed_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    viewed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    view_count INTEGER DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    UNIQUE (user_id, project_id)
                )
            """)
            
            # Create chat history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chathistory(
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context_type TEXT NULL CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL') OR context_type IS NULL),
                    context_id TEXT NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata (id)
                )
            """)
            
            # Create indexes for chathistory table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chathistory_context ON chathistory (context_type, context_id)")
            
            # Create projects table (handle migration from doc_id to doc_ids)
            # First check if projects table exists and what columns it has
            cur.execute("PRAGMA table_info(projects)")
            columns = [row[1] for row in cur.fetchall()]
            
            if not columns:
                cur.execute("""
                    CREATE TABLE projects(
                        id INTEGER PRIMARY KEY,
                        project_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        user_id INTEGER NOT NULL,
                        doc_ids TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES userdata (id)
                    )
                """)
            elif 'doc_id' in columns and 'doc_ids' not in columns:
                self._migrate_projects_schema(cur)
            elif 'doc_ids' not in columns:
                cur.execute("ALTER TABLE projects ADD COLUMN doc_ids TEXT")
            
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents(
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    vector_path TEXT NOT NULL,
                    pages INTEGER DEFAULT 0,
                    chunks_indexed INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    user_id INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id)
                )
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status)")
            
            # Create project_documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_documents(
                    id INTEGER PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
                    UNIQUE (project_id, doc_id)
                )
            """)
            
            # Create indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_documents_project_id ON project_documents (project_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_documents_doc_id ON project_documents (doc_id)")

            # Create project_members table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_members(
                    id INTEGER PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT DEFAULT 'member',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    UNIQUE (project_id, user_id)
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members (user_id)")

            # Create project_invitations table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS project_invitations(
                    id INTEGER PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    inviter_id INTEGER NOT NULL,
                    invitee_id INTEGER,
                    invitee_email TEXT NOT NULL,
                    role TEXT DEFAULT 'member',
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    responded_at DATETIME,
                    FOREIGN KEY (project_id) REFERENCES projects (project_id) ON DELETE CASCADE,
                    FOREIGN KEY (inviter_id) REFERENCES userdata (id) ON DELETE CASCADE,
                    FOREIGN KEY (invitee_id) REFERENCES userdata (id) ON DELETE SET NULL
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_invitations_invitee ON project_invitations (invitee_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_project_invitations_email ON project_invitations (invitee_email)")

            # Create notifications table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications(
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    notification_type TEXT DEFAULT 'general',
                    metadata TEXT,
                    is_read BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications (user_id, is_read)")

            # Create user OTP table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_otp_codes(
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    otp_code TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    consumed_at DATETIME,
                    FOREIGN KEY (user_id) REFERENCES userdata (id) ON DELETE CASCADE
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_user_otp_lookup ON user_otp_codes (user_id, purpose, consumed_at)")

            # Create chat_sessions table for enhanced session management
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions(
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL,
                    context_type TEXT NOT NULL CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL')),
                    context_id TEXT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for chat_sessions table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_context ON chat_sessions (user_id, context_type, context_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions (session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_activity ON chat_sessions (last_activity)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_active ON chat_sessions (user_id, is_active)")
            
            # Create test user if not exists
            cur.execute("SELECT * FROM userdata WHERE email = ?", ("test@example.com",))
            if not cur.fetchone():
                test_password = hashlib.sha256("testuser1".encode()).hexdigest()
                cur.execute(
                    "INSERT INTO userdata (firstname, lastname, email, password) VALUES (?, ?, ?, ?)",
                    ("Test", "User", "test@example.com", test_password)
                )
            
            # Create default admin user
            cur.execute("SELECT * FROM admin_users WHERE email = ?", ("admin@esticore.com",))
            if not cur.fetchone():
                admin_password = hashlib.sha256("admin123".encode()).hexdigest()
                cur.execute(
                    "INSERT INTO admin_users (username, email, password, is_super_admin) VALUES (?, ?, ?, ?)",
                    ("admin", "admin@esticore.com", admin_password, True)
                )
            
            # Create default AI models
            default_models = [
                ("GPT-4", "OpenAI", "gpt-4", 1, '{"temperature": 0.7, "max_tokens": 2000}'),
                ("GPT-3.5-Turbo", "OpenAI", "gpt-3.5-turbo", 0, '{"temperature": 0.7, "max_tokens": 2000}'),
                ("Claude-2", "Anthropic", "claude-2", 0, '{"temperature": 0.7, "max_tokens": 2000}')
            ]
            
            for name, provider, model_name, is_active, config in default_models:
                cur.execute("SELECT * FROM ai_models WHERE name = ?", (name,))
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO ai_models (name, provider, model_name, is_active, config) VALUES (?, ?, ?, ?, ?)",
                        (name, provider, model_name, is_active, config)
                    )
        
        conn.commit()
        conn.close()
        
        # Run migrations
        self._migrate_documents_schema()
        self._migrate_email_verification_schema()
        self._migrate_user_profile_schema()
        self._migrate_session_schema()
    
    def _get_placeholder(self):
        """Get the appropriate parameter placeholder for the database type"""
        if self.use_rds:
            return "%s" if not self.is_postgres else "%s"
        return "?"

    def _quote_identifier(self, identifier: str) -> str:
        """Safely quote a column or table identifier for the active database."""
        if self.use_rds:
            if self.is_postgres:
                return f'"{identifier}"'
            return f'`{identifier}`'
        return f'"{identifier}"'

    def ensure_subscription_pricing_schema(self):
        """Ensure the subscription_plan_prices table exists for the active database."""

        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            if self.use_rds and self.is_postgres:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                        id SERIAL PRIMARY KEY,
                        plan_id INTEGER NOT NULL,
                        duration_months INTEGER NOT NULL,
                        price NUMERIC(10, 2) NOT NULL,
                        currency VARCHAR(10) DEFAULT 'usd',
                        stripe_price_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (plan_id, duration_months),
                        FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_plan_prices_plan ON subscription_plan_prices (plan_id)")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_plan_prices_duration ON subscription_plan_prices (duration_months)"
                )
            elif self.use_rds:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        plan_id INT NOT NULL,
                        duration_months INT NOT NULL,
                        price DECIMAL(10,2) NOT NULL,
                        currency VARCHAR(10) DEFAULT 'usd',
                        stripe_price_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_plan_duration (plan_id, duration_months),
                        FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                    ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS subscription_plan_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plan_id INTEGER NOT NULL,
                        duration_months INTEGER NOT NULL,
                        price REAL NOT NULL,
                        currency TEXT DEFAULT 'usd',
                        stripe_price_id TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(plan_id, duration_months),
                        FOREIGN KEY (plan_id) REFERENCES subscription_plans (id) ON DELETE CASCADE
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_plan_prices_plan ON subscription_plan_prices (plan_id)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_plan_prices_duration ON subscription_plan_prices (duration_months)"
                )

            conn.commit()
        except (PostgreSQLError, sqlite3.Error) as exc:
            if conn:
                conn.rollback()
            raise Exception(f"Failed to ensure subscription pricing schema: {exc}") from exc
        finally:
            if conn:
                conn.close()

    def ensure_pgvector_extension(self) -> bool:
        """Ensure pgvector extension is available in PostgreSQL"""
        if not self.is_postgres:
            return True  # Not needed for non-PostgreSQL databases
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Try to create the extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            
            # Verify the extension is available
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            result = cur.fetchone()
            
            if result:
                print("pgvector extension is available")
                return True
            else:
                print("WARNING: pgvector extension could not be created")
                return False
                
        except Exception as e:
            print(f"Error setting up pgvector extension: {e}")
            print("Please install pgvector extension manually:")
            print("1. Connect to your PostgreSQL database as superuser")
            print("2. Run: CREATE EXTENSION vector;")
            return False
        finally:
            if conn:
                conn.close()
    
    def create_new_tables(self):
        """Create new tables for file storage and vector operations"""
        if not self.is_postgres:
            print("New table creation is only supported for PostgreSQL")
            return
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Create file_storage table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS file_storage (
                    id SERIAL PRIMARY KEY,
                    file_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    content_type VARCHAR(100) NOT NULL,
                    file_size BIGINT NOT NULL,
                    file_data BYTEA NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for file_storage
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_storage_file_id ON file_storage (file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_storage_user_id ON file_storage (user_id)")
            
            # Create vector_chunks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for vector_chunks
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_doc_id ON vector_chunks (doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_page_number ON vector_chunks (page_number)")
            
            # Create vector similarity index (only if pgvector is available)
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_embedding ON vector_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
            except Exception as e:
                print(f"Could not create vector index (pgvector may not be available): {e}")
            
            # Create generated_outputs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS generated_outputs (
                    id SERIAL PRIMARY KEY,
                    output_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    content_type VARCHAR(100) NOT NULL,
                    file_size BIGINT NOT NULL,
                    file_data BYTEA NOT NULL,
                    source_doc_id VARCHAR(255),
                    user_id INTEGER NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_doc_id) REFERENCES documents(doc_id) ON DELETE SET NULL,
                    FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for generated_outputs
            cur.execute("CREATE INDEX IF NOT EXISTS idx_generated_outputs_output_id ON generated_outputs (output_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_generated_outputs_source_doc_id ON generated_outputs (source_doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_generated_outputs_user_id ON generated_outputs (user_id)")
            
            conn.commit()
            print("Successfully created new tables for file and vector storage")
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error creating new tables: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def migrate_documents_table(self):
        """Migrate documents table to remove file paths and add file_id"""
        if not self.is_postgres:
            print("Documents table migration is only supported for PostgreSQL")
            return
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Check if file_id column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name = 'file_id'
            """)
            
            if not cur.fetchone():
                # Add file_id column
                cur.execute("ALTER TABLE documents ADD COLUMN file_id VARCHAR(255)")
                
                # Add foreign key constraint
                cur.execute("""
                    ALTER TABLE documents 
                    ADD CONSTRAINT fk_documents_file_id 
                    FOREIGN KEY (file_id) REFERENCES file_storage(file_id) ON DELETE SET NULL
                """)
                
                print("Added file_id column to documents table")
            
            # Check if old columns exist and remove them
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'documents' AND column_name IN ('pdf_path', 'vector_path')
            """)
            
            old_columns = cur.fetchall()
            for column in old_columns:
                column_name = column[0] if isinstance(column, tuple) else column['column_name']
                cur.execute(f"ALTER TABLE documents DROP COLUMN IF EXISTS {column_name}")
                print(f"Removed {column_name} column from documents table")
            
            conn.commit()
            print("Successfully migrated documents table")
            
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error migrating documents table: {e}")
            # Don't raise exception to prevent breaking initialization
        finally:
            if conn:
                conn.close()
    
    # File storage operations
    def store_file(self, file_id: str, filename: str, content_type: str, file_data: bytes, user_id: int) -> bool:
        """Store file as binary data in database"""
        if not self.is_postgres:
            raise Exception("File storage is only supported with PostgreSQL")
        
        # Validate input parameters
        if not file_id or not filename or not content_type:
            raise ValueError("file_id, filename, and content_type are required")
        if not isinstance(file_data, bytes):
            raise ValueError("file_data must be bytes")
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO file_storage (file_id, filename, content_type, file_size, file_data, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (file_id, filename, content_type, len(file_data), file_data, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error storing file: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_file(self, file_id: str) -> Optional[FileStorage]:
        """Retrieve file from database"""
        if not self.is_postgres:
            raise Exception("File storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT id, file_id, filename, content_type, file_size, file_data, user_id, created_at, updated_at
                FROM file_storage WHERE file_id = %s
            """, (file_id,))
            
            row = cur.fetchone()
            if row:
                # Convert memoryview to bytes if needed
                file_data = row['file_data']
                if isinstance(file_data, memoryview):
                    file_data = file_data.tobytes()
                
                return FileStorage(
                    id=row['id'],
                    file_id=row['file_id'],
                    filename=row['filename'],
                    content_type=row['content_type'],
                    file_size=row['file_size'],
                    file_data=file_data,
                    user_id=row['user_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            return None
            
        except Exception as e:
            raise Exception(f"Error retrieving file: {e}")
        finally:
            if conn:
                conn.close()
    
    def delete_file(self, file_id: str) -> bool:
        """Delete file from database"""
        if not self.is_postgres:
            raise Exception("File storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("DELETE FROM file_storage WHERE file_id = %s", (file_id,))
            conn.commit()
            
            return cur.rowcount > 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error deleting file: {e}")
        finally:
            if conn:
                conn.close()
    
    # Vector storage operations
    def store_vector_chunks(self, doc_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Store vector chunks for a document"""
        if not self.is_postgres:
            raise Exception("Vector storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            stored_count = 0
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO vector_chunks (chunk_id, doc_id, page_number, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    chunk_id,
                    doc_id,
                    chunk.get('page', 1),
                    chunk.get('text', ''),
                    chunk.get('embedding')
                ))
                stored_count += 1
            
            conn.commit()
            return stored_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error storing vector chunks: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_vector_chunks(self, doc_id: str) -> List[VectorChunk]:
        """Get all vector chunks for a document"""
        if not self.is_postgres:
            raise Exception("Vector storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT id, chunk_id, doc_id, page_number, chunk_text, embedding, created_at
                FROM vector_chunks WHERE doc_id = %s ORDER BY page_number, id
            """, (doc_id,))
            
            chunks = []
            for row in cur.fetchall():
                chunks.append(VectorChunk(
                    id=row['id'],
                    chunk_id=row['chunk_id'],
                    doc_id=row['doc_id'],
                    page_number=row['page_number'],
                    chunk_text=row['chunk_text'],
                    embedding=row['embedding'],
                    created_at=row['created_at']
                ))
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error retrieving vector chunks: {e}")
        finally:
            if conn:
                conn.close()
    
    def similarity_search(self, doc_id: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using pgvector"""
        if not self.is_postgres:
            raise Exception("Vector similarity search is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Convert embedding to string format for PostgreSQL
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            cur.execute("""
                SELECT chunk_id, doc_id, page_number, chunk_text, 
                       embedding <=> %s::vector as distance
                FROM vector_chunks 
                WHERE doc_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, doc_id, embedding_str, k))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'chunk_id': row['chunk_id'],
                    'doc_id': row['doc_id'],
                    'page': row['page_number'],
                    'text': row['chunk_text'],
                    'distance': float(row['distance']),
                    'similarity_score': 1 - float(row['distance'])  # Convert distance to similarity
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {e}")
        finally:
            if conn:
                conn.close()
    
    def delete_vector_chunks(self, doc_id: str) -> bool:
        """Delete all vector chunks for a document"""
        if not self.is_postgres:
            raise Exception("Vector storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("DELETE FROM vector_chunks WHERE doc_id = %s", (doc_id,))
            conn.commit()
            
            return cur.rowcount > 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error deleting vector chunks: {e}")
        finally:
            if conn:
                conn.close()
    
    # Generated output operations
    def store_generated_output(self, output_id: str, filename: str, content_type: str, 
                              file_data: bytes, source_doc_id: str, user_id: int, 
                              metadata: Dict[str, Any] = None) -> bool:
        """Store generated output file"""
        if not self.is_postgres:
            raise Exception("Generated output storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO generated_outputs (output_id, filename, content_type, file_size, 
                                             file_data, source_doc_id, user_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (output_id, filename, content_type, len(file_data), file_data, 
                  source_doc_id, user_id, json.dumps(metadata) if metadata else None))
            
            conn.commit()
            return True
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error storing generated output: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_generated_output(self, output_id: str) -> Optional[GeneratedOutput]:
        """Retrieve generated output from database"""
        if not self.is_postgres:
            raise Exception("Generated output storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT id, output_id, filename, content_type, file_size, file_data, 
                       source_doc_id, user_id, metadata, created_at
                FROM generated_outputs WHERE output_id = %s
            """, (output_id,))
            
            row = cur.fetchone()
            if row:
                # Convert memoryview to bytes if needed
                file_data = row['file_data']
                if isinstance(file_data, memoryview):
                    file_data = file_data.tobytes()
                
                return GeneratedOutput(
                    id=row['id'],
                    output_id=row['output_id'],
                    filename=row['filename'],
                    content_type=row['content_type'],
                    file_size=row['file_size'],
                    file_data=file_data,
                    source_doc_id=row['source_doc_id'],
                    user_id=row['user_id'],
                    metadata=row['metadata'] if row['metadata'] else None,  # JSONB is already parsed
                    created_at=row['created_at']
                )
            return None
            
        except Exception as e:
            raise Exception(f"Error retrieving generated output: {e}")
        finally:
            if conn:
                conn.close()
    
    def list_generated_outputs(self, user_id: int = None) -> List[GeneratedOutput]:
        """List generated outputs, optionally filtered by user"""
        if not self.is_postgres:
            raise Exception("Generated output storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if user_id:
                cur.execute("""
                    SELECT id, output_id, filename, content_type, file_size, 
                           source_doc_id, user_id, metadata, created_at
                    FROM generated_outputs WHERE user_id = %s ORDER BY created_at DESC
                """, (user_id,))
            else:
                cur.execute("""
                    SELECT id, output_id, filename, content_type, file_size, 
                           source_doc_id, user_id, metadata, created_at
                    FROM generated_outputs ORDER BY created_at DESC
                """)
            
            outputs = []
            for row in cur.fetchall():
                outputs.append(GeneratedOutput(
                    id=row['id'],
                    output_id=row['output_id'],
                    filename=row['filename'],
                    content_type=row['content_type'],
                    file_size=row['file_size'],
                    file_data=b"",  # Don't load file data for listing
                    source_doc_id=row['source_doc_id'],
                    user_id=row['user_id'],
                    metadata=row['metadata'] if row['metadata'] else None,  # JSONB is already parsed
                    created_at=row['created_at']
                ))
            
            return outputs
            
        except Exception as e:
            raise Exception(f"Error listing generated outputs: {e}")
        finally:
            if conn:
                conn.close()
    
    def delete_generated_output(self, output_id: str) -> bool:
        """Delete generated output from database"""
        if not self.is_postgres:
            raise Exception("Generated output storage is only supported with PostgreSQL")
        
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("DELETE FROM generated_outputs WHERE output_id = %s", (output_id,))
            conn.commit()
            
            return cur.rowcount > 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error deleting generated output: {e}")
        finally:
            if conn:
                conn.close()
    
    def _migrate_projects_schema(self, cur):
        """Migrate projects table from doc_id to doc_ids schema"""
        cur.execute("""
            CREATE TABLE projects_new(
                id INTEGER PRIMARY KEY,
                project_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                user_id INTEGER NOT NULL,
                doc_ids TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES userdata (id)
            )
        """)
        
        cur.execute("""
            INSERT INTO projects_new (id, project_id, name, description, user_id, doc_ids, created_at, updated_at)
            SELECT id, project_id, name, description, user_id, 
                   CASE 
                       WHEN doc_id IS NOT NULL THEN '["' || doc_id || '"]'
                       ELSE NULL
                   END as doc_ids,
                   created_at, updated_at
            FROM projects
        """)
        
        cur.execute("DROP TABLE projects")
        cur.execute("ALTER TABLE projects_new RENAME TO projects")
    
    def _migrate_documents_schema(self):
        """Migrate documents table to include vector_path column if it doesn't exist"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            if self.use_rds:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = %s 
                    AND TABLE_NAME = 'documents'
                """, (settings.DB_NAME,))
                table_exists = cur.fetchone()[0] > 0
                
                if table_exists:
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s 
                        AND TABLE_NAME = 'documents' 
                        AND COLUMN_NAME = 'vector_path'
                    """, (settings.DB_NAME,))
                    column_exists = cur.fetchone()[0] > 0
                    
                    if not column_exists:
                        cur.execute("ALTER TABLE documents ADD COLUMN vector_path TEXT")
                        cur.execute("SELECT doc_id FROM documents WHERE vector_path IS NULL")
                        docs_to_update = cur.fetchall()
                        
                        for (doc_id,) in docs_to_update:
                            vector_path = os.path.join(settings.VECTORS_DIR, doc_id)
                            cur.execute("UPDATE documents SET vector_path = %s WHERE doc_id = %s", (vector_path, doc_id))
                        
                        conn.commit()
            else:
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
                table_exists = cur.fetchone() is not None
                
                if table_exists:
                    cur.execute("PRAGMA table_info(documents)")
                    columns = [row[1] for row in cur.fetchall()]
                    
                    if 'vector_path' not in columns:
                        cur.execute("ALTER TABLE documents ADD COLUMN vector_path TEXT NOT NULL DEFAULT ''")
                        cur.execute("SELECT doc_id FROM documents WHERE vector_path = '' OR vector_path IS NULL")
                        docs_to_update = cur.fetchall()
                        
                        for (doc_id,) in docs_to_update:
                            vector_path = os.path.join(settings.VECTORS_DIR, doc_id)
                            cur.execute("UPDATE documents SET vector_path = ? WHERE doc_id = ?", (vector_path, doc_id))
                        
                        conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def _migrate_email_verification_schema(self):
        """Add email verification columns to existing userdata table"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            if self.use_rds:
                if self.is_postgres:
                    # Check if email verification columns exist in PostgreSQL
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.columns 
                        WHERE table_name = 'userdata' 
                        AND column_name = 'is_verified'
                    """)
                    
                    column_exists = cur.fetchone()[0] > 0
                    
                    if not column_exists:
                        print("Adding email verification columns to userdata table (PostgreSQL)...")
                        cur.execute("ALTER TABLE userdata ADD COLUMN is_verified BOOLEAN DEFAULT FALSE")
                        cur.execute("ALTER TABLE userdata ADD COLUMN verification_token VARCHAR(255)")
                        cur.execute("ALTER TABLE userdata ADD COLUMN verification_token_expires TIMESTAMP NULL")
                        
                        # Set Google OAuth users as verified by default
                        cur.execute("UPDATE userdata SET is_verified = TRUE WHERE google_id IS NOT NULL")
                        
                        conn.commit()
                        print("Email verification columns added successfully")
                    else:
                        print("Email verification columns already exist in userdata table")
                else:
                    # MySQL logic
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s 
                        AND TABLE_NAME = 'userdata' 
                        AND COLUMN_NAME = 'is_verified'
                    """, (settings.DB_NAME,))
                    
                    column_exists = cur.fetchone()[0] > 0
                    
                    if not column_exists:
                        print("Adding email verification columns to userdata table (MySQL)...")
                        cur.execute("ALTER TABLE userdata ADD COLUMN is_verified BOOLEAN DEFAULT FALSE")
                        cur.execute("ALTER TABLE userdata ADD COLUMN verification_token VARCHAR(255)")
                        cur.execute("ALTER TABLE userdata ADD COLUMN verification_token_expires TIMESTAMP NULL")
                        
                        # Set Google OAuth users as verified by default
                        cur.execute("UPDATE userdata SET is_verified = TRUE WHERE google_id IS NOT NULL")
                        
                        conn.commit()
                        print("Email verification columns added successfully")
                    else:
                        print("Email verification columns already exist in userdata table")
            else:
                cur.execute("PRAGMA table_info(userdata)")
                columns = [row[1] for row in cur.fetchall()]
                
                if 'is_verified' not in columns:
                    print("Adding email verification columns to userdata table (SQLite)...")
                    cur.execute("ALTER TABLE userdata ADD COLUMN is_verified BOOLEAN DEFAULT 0")
                    cur.execute("ALTER TABLE userdata ADD COLUMN verification_token VARCHAR(255)")
                    cur.execute("ALTER TABLE userdata ADD COLUMN verification_token_expires DATETIME")
                    cur.execute("UPDATE userdata SET is_verified = 1 WHERE google_id IS NOT NULL")
                    conn.commit()
                    print("Email verification columns added successfully")
                else:
                    print("Email verification columns already exist in userdata table")
                    
        except Exception as e:
            print(f"Email verification schema migration error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _migrate_user_profile_schema(self):
        """Ensure profile-related columns exist on userdata table"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            if self.use_rds:
                if self.is_postgres:
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'userdata'
                        """
                    )
                    columns = {row[0] for row in cur.fetchall()}

                    statements = []
                    if 'is_active' not in columns:
                        statements.append("ALTER TABLE userdata ADD COLUMN is_active BOOLEAN DEFAULT TRUE")
                    if 'profile_image' not in columns:
                        statements.append("ALTER TABLE userdata ADD COLUMN profile_image TEXT")
                    if 'role' not in columns:
                        statements.append("ALTER TABLE userdata ADD COLUMN role VARCHAR(50) DEFAULT 'user'")
                    if 'last_login' not in columns:
                        statements.append("ALTER TABLE userdata ADD COLUMN last_login TIMESTAMP NULL")

                    for statement in statements:
                        cur.execute(statement)

                    if statements:
                        conn.commit()
                else:
                    cur.execute(
                        """
                        SELECT COLUMN_NAME
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = %s
                        AND TABLE_NAME = 'userdata'
                        """,
                        (settings.DB_NAME,)
                    )
                    columns = {row[0] for row in cur.fetchall()}

                    def ensure(column: str, definition: str):
                        if column not in columns:
                            cur.execute(f"ALTER TABLE userdata ADD COLUMN {definition}")
                            columns.add(column)

                    ensure('is_active', 'BOOLEAN DEFAULT TRUE')
                    ensure('profile_image', 'TEXT')
                    ensure('role', "VARCHAR(50) DEFAULT 'user'")
                    ensure('last_login', 'TIMESTAMP NULL')

                    conn.commit()
            else:
                cur.execute("PRAGMA table_info(userdata)")
                columns = {row[1] for row in cur.fetchall()}

                def ensure_sqlite(column: str, definition: str):
                    if column not in columns:
                        cur.execute(f"ALTER TABLE userdata ADD COLUMN {definition}")
                        columns.add(column)

                ensure_sqlite('is_active', 'BOOLEAN DEFAULT 1')
                ensure_sqlite('profile_image', 'TEXT')
                ensure_sqlite('role', "TEXT DEFAULT 'user'")
                ensure_sqlite('last_login', 'DATETIME')

                conn.commit()
        except Exception as e:
            print(f"User profile schema migration error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _migrate_session_schema(self):
        """Migrate existing tables to support enhanced session management"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            if self.use_rds:
                if self.is_postgres:
                    # PostgreSQL migration logic
                    # Check if chat_sessions table exists
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.tables 
                        WHERE table_name = 'chat_sessions'
                    """)
                    
                    table_exists = cur.fetchone()[0] > 0
                    
                    if not table_exists:
                        print("Creating chat_sessions table (PostgreSQL)...")
                        # Table is already created in init_database for PostgreSQL
                        print("chat_sessions table already created in init_database")
                    
                    # Check if context columns exist in chathistory table
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.columns 
                        WHERE table_name = 'chathistory' 
                        AND column_name = 'context_type'
                    """)
                    
                    context_columns_exist = cur.fetchone()[0] > 0
                    
                    if not context_columns_exist:
                        print("Adding context columns to chathistory table (PostgreSQL)...")
                        cur.execute("ALTER TABLE chathistory ADD COLUMN context_type VARCHAR(20) CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL'))")
                        cur.execute("ALTER TABLE chathistory ADD COLUMN context_id VARCHAR(255)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_chathistory_context ON chathistory (context_type, context_id)")
                        conn.commit()
                        print("Context columns added to chathistory table successfully")
                else:
                    # MySQL migration logic
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_SCHEMA = %s 
                        AND TABLE_NAME = 'chat_sessions'
                    """, (settings.DB_NAME,))
                    
                    table_exists = cur.fetchone()[0] > 0
                    
                    if not table_exists:
                        print("Creating chat_sessions table (MySQL)...")
                        cur.execute("""
                            CREATE TABLE chat_sessions(
                                id INT AUTO_INCREMENT PRIMARY KEY,
                                session_id VARCHAR(255) UNIQUE NOT NULL,
                                user_id INT NOT NULL,
                                context_type ENUM('PROJECT', 'DOCUMENT', 'GENERAL') NOT NULL,
                                context_id VARCHAR(255) NULL,
                                is_active BOOLEAN DEFAULT TRUE,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                                metadata JSON NULL,
                                FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE,
                                INDEX idx_user_context (user_id, context_type, context_id),
                                INDEX idx_session_id (session_id),
                                INDEX idx_last_activity (last_activity),
                                INDEX idx_active_sessions (user_id, is_active)
                            ) ENGINE=InnoDB CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                        """)
                        conn.commit()
                        print("chat_sessions table created successfully")
                    
                    # Check if context columns exist in chathistory table
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s 
                        AND TABLE_NAME = 'chathistory' 
                        AND COLUMN_NAME = 'context_type'
                    """, (settings.DB_NAME,))
                    
                    context_columns_exist = cur.fetchone()[0] > 0
                    
                    if not context_columns_exist:
                        print("Adding context columns to chathistory table (MySQL)...")
                        cur.execute("ALTER TABLE chathistory ADD COLUMN context_type ENUM('PROJECT', 'DOCUMENT', 'GENERAL') NULL")
                        cur.execute("ALTER TABLE chathistory ADD COLUMN context_id VARCHAR(255) NULL")
                        cur.execute("CREATE INDEX idx_chathistory_context ON chathistory (context_type, context_id)")
                        conn.commit()
                        print("Context columns added to chathistory table successfully")
                    
            else:
                # Check if chat_sessions table exists in SQLite
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_sessions'")
                table_exists = cur.fetchone() is not None
                
                if not table_exists:
                    print("Creating chat_sessions table (SQLite)...")
                    cur.execute("""
                        CREATE TABLE chat_sessions(
                            id INTEGER PRIMARY KEY,
                            session_id TEXT UNIQUE NOT NULL,
                            user_id INTEGER NOT NULL,
                            context_type TEXT NOT NULL CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL')),
                            context_id TEXT NULL,
                            is_active BOOLEAN DEFAULT 1,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT NULL,
                            FOREIGN KEY (user_id) REFERENCES userdata(id) ON DELETE CASCADE
                        )
                    """)
                    
                    # Create indexes
                    cur.execute("CREATE INDEX idx_chat_sessions_user_context ON chat_sessions (user_id, context_type, context_id)")
                    cur.execute("CREATE INDEX idx_chat_sessions_session_id ON chat_sessions (session_id)")
                    cur.execute("CREATE INDEX idx_chat_sessions_last_activity ON chat_sessions (last_activity)")
                    cur.execute("CREATE INDEX idx_chat_sessions_active ON chat_sessions (user_id, is_active)")
                    
                    conn.commit()
                    print("chat_sessions table created successfully")
                
                # Check if context columns exist in chathistory table
                cur.execute("PRAGMA table_info(chathistory)")
                columns = [row[1] for row in cur.fetchall()]
                
                if 'context_type' not in columns:
                    print("Adding context columns to chathistory table (SQLite)...")
                    cur.execute("ALTER TABLE chathistory ADD COLUMN context_type TEXT NULL CHECK (context_type IN ('PROJECT', 'DOCUMENT', 'GENERAL') OR context_type IS NULL)")
                    cur.execute("ALTER TABLE chathistory ADD COLUMN context_id TEXT NULL")
                    cur.execute("CREATE INDEX idx_chathistory_context ON chathistory (context_type, context_id)")
                    conn.commit()
                    print("Context columns added to chathistory table successfully")
                    
        except Exception as e:
            print(f"Session schema migration error: {e}")
            if conn:
                conn.rollback()
            # Don't raise the exception to prevent breaking initialization
        finally:
            if conn:
                conn.close()

    # User management methods
    def create_user(self, firstname: str, lastname: str, email: str, password: str, google_id: str = None) -> int:
        """Create a new user"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            insert_sql = (
                f"INSERT INTO userdata (firstname, lastname, email, password, google_id) "
                f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
            )
            params = (firstname, lastname, email, hashed_password, google_id)

            cur.execute(
                insert_sql,
                params
            )
            conn.commit()
            
            cur.execute(f"SELECT id FROM userdata WHERE email = {placeholder}", (email,))
            user = cur.fetchone()
            return user[0] if user else None
        except PostgreSQLError as e:
            # Handle possible sequence desync causing duplicate primary key
            # Error code 23505 = unique_violation
            if self.use_rds and self.is_postgres and getattr(e, 'pgcode', None) == '23505':
                try:
                    conn.rollback()
                    # Sync sequence to current MAX(id)
                    cur.execute(
                        "SELECT setval(pg_get_serial_sequence('userdata','id'), "
                        "COALESCE((SELECT MAX(id) FROM userdata), 1), "
                        "((SELECT MAX(id) FROM userdata) IS NOT NULL))"
                    )
                    conn.commit()
                    # Retry once
                    cur.execute(insert_sql, params)
                    conn.commit()
                    cur.execute(f"SELECT id FROM userdata WHERE email = {placeholder}", (email,))
                    user = cur.fetchone()
                    return user[0] if user else None
                except Exception:
                    conn.rollback()
                    raise
            raise
        finally:
            conn.close()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, firstname, lastname, email, password, google_id, is_verified, verification_token, "
                f"verification_token_expires, created_at, is_active, profile_image, role, last_login "
                f"FROM userdata WHERE email = {placeholder}",
                (email,)
            )
            row = cur.fetchone()

            return self._map_user_row(row)
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, firstname, lastname, email, password, google_id, is_verified, verification_token, "
                f"verification_token_expires, created_at, is_active, profile_image, role, last_login "
                f"FROM userdata WHERE id = {placeholder}",
                (user_id,)
            )
            row = cur.fetchone()

            return self._map_user_row(row)
        finally:
            conn.close()
    
    def verify_user_credentials(self, email: str, password: str) -> Optional[User]:
        """Verify user credentials"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, firstname, lastname, email, password, google_id, is_verified, verification_token, "
                f"verification_token_expires, created_at, is_active, profile_image, role, last_login "
                f"FROM userdata WHERE email = {placeholder} AND password = {placeholder}",
                (email, hashed_password)
            )
            row = cur.fetchone()

            return self._map_user_row(row)
        finally:
            conn.close()

    def get_user_by_google_id(self, google_id: str) -> Optional[User]:
        """Get user by Google ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, firstname, lastname, email, password, google_id, is_verified, verification_token, "
                f"verification_token_expires, created_at, is_active, profile_image, role, last_login "
                f"FROM userdata WHERE google_id = {placeholder}",
                (google_id,)
            )
            row = cur.fetchone()

            return self._map_user_row(row)
        finally:
            conn.close()
    
    def update_user_google_id(self, user_id: int, google_id: str):
        """Update user's Google ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(f"UPDATE userdata SET google_id = {placeholder} WHERE id = {placeholder}", (google_id, user_id))
            conn.commit()
        finally:
            conn.close()

    def update_user_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()

        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"UPDATE userdata SET password = {placeholder} WHERE id = {placeholder}",
                (hashed_password, user_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def update_user_profile(self, user_id: int, **kwargs) -> bool:
        """Update user profile"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            update_fields = []
            params = []

            for key, value in kwargs.items():
                if value is not None:
                    update_fields.append(f"{key} = {placeholder}")
                    params.append(value)

            if not update_fields:
                return False

            params.append(user_id)
            query = f"UPDATE userdata SET {', '.join(update_fields)} WHERE id = {placeholder}"
            cur.execute(query, params)
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def update_user_status(self, user_id: int, is_active: bool) -> bool:
        """Update user active status"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"UPDATE userdata SET is_active = {placeholder} WHERE id = {placeholder}",
                (is_active, user_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user and all their data"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"DELETE FROM userdata WHERE id = {placeholder}", (user_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def get_all_users(self) -> List[User]:
        """Get all users in the system"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT id, firstname, lastname, email, password, google_id, is_verified,
                       verification_token, verification_token_expires, created_at, is_active, profile_image,
                       role, last_login
                FROM userdata
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()

            return [self._map_user_row(row) for row in rows if row]
        finally:
            conn.close()

    def get_all_users_paginated(self, page: int = 1, limit: int = 20, status: Optional[str] = None, search: Optional[str] = None) -> List[User]:
        """Get users with pagination and filtering"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            # Base query
            query = """
                SELECT id, firstname, lastname, email, password, google_id, is_verified,
                    verification_token, verification_token_expires, created_at, is_active, profile_image,
                    role, last_login
                FROM userdata
                WHERE 1=1
            """
            params = []
            
            # Add status filter
            if status:
                if status == "active":
                    query += f" AND is_active = {placeholder}"
                    params.append(True)
                elif status == "inactive": 
                    query += f" AND is_active = {placeholder}"
                    params.append(False)
            
            # Add search filter
            if search:
                query += f" AND (email LIKE {placeholder} OR firstname LIKE {placeholder} OR lastname LIKE {placeholder})"
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term])
            
            # Add ordering and pagination
            query += f" ORDER BY created_at DESC LIMIT {placeholder} OFFSET {placeholder}"
            
            # Calculate offset
            offset = (page - 1) * limit
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            return [self._map_user_row(row) for row in rows if row]
        finally:
            conn.close()

    def get_all_users_with_filters(self, status: Optional[str] = None, search: Optional[str] = None) -> List[User]:
        """Get all users applying optional status/search filters"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            query = """
                SELECT id, firstname, lastname, email, password, google_id, is_verified,
                    verification_token, verification_token_expires, created_at, is_active, profile_image,
                    role, last_login
                FROM userdata
                WHERE 1=1
            """
            params: List[Any] = []

            if status:
                if status == "active":
                    query += f" AND is_active = {placeholder}"
                    params.append(True)
                elif status == "inactive":
                    query += f" AND is_active = {placeholder}"
                    params.append(False)

            if search:
                query += f" AND (email LIKE {placeholder} OR firstname LIKE {placeholder} OR lastname LIKE {placeholder})"
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term])

            query += " ORDER BY created_at DESC"

            cur.execute(query, params)
            rows = cur.fetchall()
            return [self._map_user_row(row) for row in rows if row]
        finally:
            conn.close()

    

    # Admin management methods
    def create_admin_user(self, username: str, email: str, password: str, is_super_admin: bool = False) -> int:
        """Create a new admin user"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"INSERT INTO admin_users (username, email, password, is_super_admin) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (username, email, hashed_password, is_super_admin)
            )
            conn.commit()
            
            cur.execute(f"SELECT id FROM admin_users WHERE email = {placeholder}", (email,))
            admin = cur.fetchone()
            return admin[0] if admin else None
        finally:
            conn.close()
    
    def get_admin_by_email(self, email: str) -> Optional[AdminUser]:
        """Get admin by email"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, username, email, password, is_super_admin, created_at, last_login FROM admin_users WHERE email = {placeholder}",
                (email,)
            )
            row = cur.fetchone()
            
            if row:
                return AdminUser(
                    id=row[0], username=row[1], email=row[2], password=row[3],
                    is_super_admin=bool(row[4]), created_at=row[5], last_login=row[6]
                )
            return None
        finally:
            conn.close()
    
    def verify_admin_credentials(self, email: str, password: str) -> Optional[AdminUser]:
        """Verify admin credentials"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, username, email, password, is_super_admin, created_at, last_login FROM admin_users WHERE email = {placeholder} AND password = {placeholder}",
                (email, hashed_password)
            )
            row = cur.fetchone()
            
            if row:
                return AdminUser(
                    id=row[0], username=row[1], email=row[2], password=row[3],
                    is_super_admin=bool(row[4]), created_at=row[5], last_login=row[6]
                )
            return None
        finally:
            conn.close()
    
    def update_admin_last_login(self, admin_id: int):
        """Update admin last login timestamp"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.use_rds:
                cur.execute(f"UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE id = {placeholder}", (admin_id,))
            else:
                cur.execute(f"UPDATE admin_users SET last_login = datetime('now') WHERE id = {placeholder}", (admin_id,))
            conn.commit()
        finally:
            conn.close()
    
    def get_super_admins(self) -> List[AdminUser]:
        """Get all super admins"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT id, username, email, password, is_super_admin, created_at, last_login FROM admin_users WHERE is_super_admin = TRUE")
            rows = cur.fetchall()
            
            return [
                AdminUser(
                    id=row[0], username=row[1], email=row[2], password=row[3],
                    is_super_admin=bool(row[4]), created_at=row[5], last_login=row[6]
                )
                for row in rows
            ]
        finally:
            conn.close()

    # Subscription plan methods
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
        prices: List[Dict[str, Any]],
    ) -> int:
        """Create a new subscription plan with normalized pricing."""

        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            features_json = self._serialize_features(features)
            cur.execute(
                f"INSERT INTO subscription_plans (name, description, storage_gb, project_limit, user_limit, action_limit, features, has_free_trial, trial_days) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (
                    name,
                    description,
                    storage_gb,
                    project_limit,
                    user_limit,
                    action_limit,
                    features_json,
                    has_free_trial,
                    trial_days,
                ),
            )
            conn.commit()

            if self.use_rds:
                plan_id = cur.lastrowid
            else:
                plan_id = cur.lastrowid

            if not plan_id:
                cur.execute(f"SELECT id FROM subscription_plans WHERE name = {placeholder}", (name,))
                row = cur.fetchone()
                plan_id = row[0] if row else None

            if not plan_id:
                raise ValueError("Failed to create subscription plan")

            normalized_prices = self._normalize_plan_prices(prices)
            if normalized_prices:
                self._upsert_subscription_plan_prices(cur, plan_id, normalized_prices)
                conn.commit()

            return plan_id
        finally:
            conn.close()

    def _serialize_features(self, features: Optional[List[str]]) -> str:
        if not features:
            return "[]"
        try:
            return json.dumps(features)
        except TypeError:
            return json.dumps(list(features))

    def _normalize_plan_prices(self, prices: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not prices:
            return normalized

        for price in prices:
            if not price:
                continue
            duration = int(price.get("duration_months", 0))
            if duration not in (6, 12):
                continue
            try:
                amount = float(price.get("price", 0))
            except (TypeError, ValueError):
                continue
            normalized.append(
                {
                    "duration_months": duration,
                    "price": amount,
                    "currency": (price.get("currency") or "usd").lower(),
                    "stripe_price_id": price.get("stripe_price_id"),
                }
            )

        return normalized

    def _upsert_subscription_plan_prices(
        self,
        cursor,
        plan_id: int,
        prices: List[Dict[str, Any]],
    ) -> None:
        placeholder = self._get_placeholder()
        cursor.execute(
            f"SELECT id, duration_months FROM subscription_plan_prices WHERE plan_id = {placeholder}",
            (plan_id,),
        )
        existing = {row[1]: row[0] for row in cursor.fetchall()}

        seen: set[int] = set()
        for price in prices:
            duration = price["duration_months"]
            seen.add(duration)
            params = (
                price["price"],
                price["currency"],
                price.get("stripe_price_id"),
            )
            if duration in existing:
                price_id = existing[duration]
                cursor.execute(
                    f"UPDATE subscription_plan_prices SET price = {placeholder}, currency = {placeholder}, stripe_price_id = {placeholder} WHERE id = {placeholder}",
                    (*params, price_id),
                )
            else:
                cursor.execute(
                    f"INSERT INTO subscription_plan_prices (plan_id, duration_months, price, currency, stripe_price_id) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                    (plan_id, duration, *params),
                )

        for duration, price_id in existing.items():
            if duration not in seen:
                cursor.execute(
                    f"DELETE FROM subscription_plan_prices WHERE id = {placeholder}",
                    (price_id,),
                )

    def _map_subscription_plan(self, row, columns: List[str]) -> SubscriptionPlan:
        col_idx = {name: idx for idx, name in enumerate(columns)}

        def get(col_name, default=None):
            idx = col_idx.get(col_name)
            if idx is None:
                return default
            try:
                return row[idx]
            except Exception:
                return default

        raw_features = get("features")
        features: List[str] = []
        if raw_features:
            try:
                if isinstance(raw_features, str):
                    features = json.loads(raw_features)
                elif isinstance(raw_features, (list, dict)):
                    features = raw_features
                elif isinstance(raw_features, (bytes, memoryview)):
                    features = json.loads(bytes(raw_features).decode("utf-8"))
            except Exception:
                features = []

        return SubscriptionPlan(
            id=get("id"),
            name=get("name"),
            description=get("description"),
            storage_gb=get("storage_gb") or 0,
            project_limit=get("project_limit") or 0,
            user_limit=get("user_limit") or 1,
            action_limit=get("action_limit") or 0,
            features=features,
            is_active=bool(get("is_active", True)),
            has_free_trial=bool(get("has_free_trial", False)),
            trial_days=get("trial_days") or 0,
            created_at=get("created_at"),
            prices=[],
        )

    def get_all_subscription_plans(self) -> List[SubscriptionPlan]:
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute(
                "SELECT id, name, description, storage_gb, project_limit, user_limit, action_limit, features, is_active, has_free_trial, trial_days, created_at FROM subscription_plans"
            )
            rows = cur.fetchall()
            columns = [
                col[0] if isinstance(col, (list, tuple)) else getattr(col, "name", col)
                for col in (cur.description or [])
            ]

            plans: List[SubscriptionPlan] = []
            for row in rows:
                plan = self._map_subscription_plan(row, columns)
                plan.prices = self.get_plan_prices(plan.id)
                if not plan.prices:
                    plan.prices = self._legacy_plan_prices(row, columns)
                plans.append(plan)

            plans.sort(key=lambda p: p.prices[0].price if p.prices else 0.0)
            return plans
        finally:
            conn.close()

    def _legacy_plan_prices(self, row, columns: List[str]) -> List[SubscriptionPlanPrice]:
        col_idx = {name: idx for idx, name in enumerate(columns)}

        def get(col_name):
            idx = col_idx.get(col_name)
            if idx is None:
                return None
            try:
                return row[idx]
            except Exception:
                return None

        prices: List[SubscriptionPlanPrice] = []
        quarterly = get("price_quarterly")
        if quarterly is not None:
            try:
                prices.append(
                    SubscriptionPlanPrice(
                        duration_months=6,
                        price=float(quarterly or 0.0),
                    )
                )
            except (TypeError, ValueError):
                pass
        annual = get("price_annual")
        if annual is not None:
            try:
                prices.append(
                    SubscriptionPlanPrice(
                        duration_months=12,
                        price=float(annual or 0.0),
                    )
                )
            except (TypeError, ValueError):
                pass
        return prices

    def get_plan_prices(self, plan_id: int) -> List[SubscriptionPlanPrice]:
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, plan_id, duration_months, price, currency, stripe_price_id, created_at FROM subscription_plan_prices WHERE plan_id = {placeholder} ORDER BY duration_months",
                (plan_id,),
            )
            rows = cur.fetchall()
            return [
                SubscriptionPlanPrice(
                    id=row[0],
                    plan_id=row[1],
                    duration_months=row[2],
                    price=float(row[3]),
                    currency=row[4] or "usd",
                    stripe_price_id=row[5],
                    created_at=row[6],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_plan_price_option(self, plan_id: int, interval: str) -> Optional[SubscriptionPlanPrice]:
        try:
            normalized = SubscriptionInterval.normalize(interval)
        except ValueError:
            return None

        months = normalized.duration_months
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, plan_id, duration_months, price, currency, stripe_price_id, created_at FROM subscription_plan_prices WHERE plan_id = {placeholder} AND duration_months = {placeholder}",
                (plan_id, months),
            )
            row = cur.fetchone()
            if not row:
                return None
            return SubscriptionPlanPrice(
                id=row[0],
                plan_id=row[1],
                duration_months=row[2],
                price=float(row[3]),
                currency=row[4] or "usd",
                stripe_price_id=row[5],
                created_at=row[6],
            )
        finally:
            conn.close()

    def get_subscription_plan_by_id(self, plan_id: int) -> Optional[SubscriptionPlan]:
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, name, description, storage_gb, project_limit, user_limit, action_limit, features, is_active, has_free_trial, trial_days, created_at FROM subscription_plans WHERE id = {placeholder}",
                (plan_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            columns = [
                col[0] if isinstance(col, (list, tuple)) else getattr(col, "name", col)
                for col in (cur.description or [])
            ]
            plan = self._map_subscription_plan(row, columns)
            plan.prices = self.get_plan_prices(plan.id)
            if not plan.prices:
                plan.prices = self._legacy_plan_prices(row, columns)
            return plan
        finally:
            conn.close()

    def get_subscription_plan_by_name(self, name: str) -> Optional[SubscriptionPlan]:
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            query = (
                "SELECT id, name, description, storage_gb, project_limit, user_limit, action_limit, features, is_active, has_free_trial, trial_days, created_at FROM subscription_plans WHERE name = %s"
                if self.use_rds
                else "SELECT id, name, description, storage_gb, project_limit, user_limit, action_limit, features, is_active, has_free_trial, trial_days, created_at FROM subscription_plans WHERE name = ?"
            )
            cur.execute(query, (name,))
            row = cur.fetchone()
            if not row:
                return None

            columns = [
                col[0] if isinstance(col, (list, tuple)) else getattr(col, "name", col)
                for col in (cur.description or [])
            ]
            plan = self._map_subscription_plan(row, columns)
            plan.prices = self.get_plan_prices(plan.id)
            if not plan.prices:
                plan.prices = self._legacy_plan_prices(row, columns)
            return plan
        finally:
            conn.close()

    def update_subscription_plan(self, plan_id: int, **kwargs) -> bool:
        prices = kwargs.pop("prices", None)

        if not kwargs and prices is None:
            return False

        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            update_fields = []
            params = []

            for key, value in kwargs.items():
                if value is None:
                    continue
                if key == "features" and isinstance(value, list):
                    value = json.dumps(value)
                update_fields.append(f"{key} = {placeholder}")
                params.append(value)

            if update_fields:
                params.append(plan_id)
                cur.execute(
                    f"UPDATE subscription_plans SET {', '.join(update_fields)} WHERE id = {placeholder}",
                    tuple(params),
                )

            if prices is not None:
                normalized = self._normalize_plan_prices(prices)
                self._upsert_subscription_plan_prices(cur, plan_id, normalized)

            conn.commit()
            return True
        finally:
            conn.close()
    
    def delete_subscription_plan(self, plan_id: int) -> bool:
        """Delete a subscription plan from the database"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()  # This should be "?" for SQLite or "%s" for PostgreSQL
        
        try:
            # The key is to properly pass the plan_id as a parameter tuple
            cur.execute(f"DELETE FROM subscription_plans WHERE id = {placeholder}", (plan_id,))
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    # Add this debug method to your DatabaseManager
    def debug_plan_features(self, plan_id: int):    
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, name, features, pg_typeof(features) as data_type FROM subscription_plans WHERE id = %s", (plan_id,))
            row = cur.fetchone()
            if row:
                print(f"Plan {row[0]} ({row[1]}):")
                print(f"  Raw features: {repr(row[2])}")
                print(f"  Data type: {row[3]}")
                print(f"  Is None: {row[2] is None}")
                print(f"  Is empty string: {row[2] == ''}")
        finally:
            conn.close()

    # User storage methods
    def create_user_storage(self, user_id: int) -> bool:
        """Create user storage record"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"INSERT INTO user_storage (user_id, used_storage_mb) VALUES ({placeholder}, 0)",
                (user_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            return False
        finally:
            conn.close()
    
    def get_user_storage(self, user_id: int) -> Optional[UserStorage]:
        """Get user storage"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, user_id, used_storage_mb, last_updated FROM user_storage WHERE user_id = {placeholder}",
                (user_id,)
            )
            row = cur.fetchone()
            
            if row:
                return UserStorage(
                    id=row[0], user_id=row[1], used_storage_mb=row[2], last_updated=row[3]
                )
            return None
        finally:
            conn.close()
    
    def update_user_storage(self, user_id: int, used_storage_mb: int) -> bool:
        """Update user storage"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            if self.use_rds:
                cur.execute(
                    f"UPDATE user_storage SET used_storage_mb = {placeholder}, last_updated = CURRENT_TIMESTAMP WHERE user_id = {placeholder}",
                    (used_storage_mb, user_id)
                )
            else:
                cur.execute(
                    f"UPDATE user_storage SET used_storage_mb = {placeholder}, last_updated = datetime('now') WHERE user_id = {placeholder}",
                    (used_storage_mb, user_id)
                )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def update_user_last_login(self, user_id: int) -> bool:
        """Update user's last login timestamp"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            if self.use_rds:
                cur.execute(
                    f"UPDATE userdata SET last_login = CURRENT_TIMESTAMP WHERE id = {placeholder}",
                    (user_id,)
                )
            else:
                cur.execute(
                    f"UPDATE userdata SET last_login = datetime('now') WHERE id = {placeholder}",
                    (user_id,)
                )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def log_user_activity(self, user_id: int, action: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a user activity entry"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            metadata_json = json.dumps(metadata) if metadata is not None else None
            cur.execute(
                f"INSERT INTO user_activity (user_id, action, metadata) VALUES ({placeholder}, {placeholder}, {placeholder})",
                (user_id, action, metadata_json)
            )
            conn.commit()
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error logging user activity: {e}")
            return False
        finally:
            conn.close()

    def get_user_activities(self, user_id: int, limit: int = 50) -> List[UserActivity]:
        """Retrieve recent user activities"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, user_id, action, metadata, created_at FROM user_activity WHERE user_id = {placeholder} ORDER BY created_at DESC LIMIT {placeholder}",
                (user_id, limit)
            )
            rows = cur.fetchall()

            activities: List[UserActivity] = []
            for row in rows:
                metadata_raw = row[3]
                try:
                    metadata_value = json.loads(metadata_raw) if metadata_raw else None
                except json.JSONDecodeError:
                    metadata_value = metadata_raw

                activities.append(
                    UserActivity(
                        id=row[0],
                        user_id=row[1],
                        action=row[2],
                        metadata=metadata_value,
                        created_at=row[4],
                    )
                )

            return activities
        finally:
            conn.close()

    # Feedback methods
    def create_feedback(self, user_id: int, email: str, ai_response: str, rating: str, project_name: str = None) -> int:
        """Create feedback record"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"INSERT INTO feedback (user_id, email, ai_response, rating, project_name) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (user_id, email, ai_response, rating, project_name)
            )
            conn.commit()
            
            cur.execute(f"SELECT id FROM feedback WHERE user_id = {placeholder} ORDER BY created_at DESC LIMIT 1", (user_id,))
            feedback = cur.fetchone()
            return feedback[0] if feedback else None
        finally:
            conn.close()
    
    def get_all_feedback(self, page: int = 1, limit: int = 20, rating: str = None) -> List[Feedback]:
        """Get all feedback with pagination"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            offset = (page - 1) * limit
            query = "SELECT id, user_id, email, ai_response, rating, project_name, created_at FROM feedback"
            params = []
            
            if rating:
                query += f" WHERE rating = {placeholder}"
                params.append(rating)
            
            query += f" ORDER BY created_at DESC LIMIT {placeholder} OFFSET {placeholder}"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            return [
                Feedback(
                    id=row[0], user_id=row[1], email=row[2], ai_response=row[3],
                    rating=row[4], project_name=row[5], created_at=row[6]
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_feedback_statistics(self) -> Dict[str, int]:
        """Get feedback statistics"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
            rows = cur.fetchall()
            
            stats = {'total': 0, 'positive': 0, 'negative': 0}
            for row in rows:
                stats[row[0]] = row[1]
                stats['total'] += row[1]
            
            return stats
        finally:
            conn.close()

    # AI model methods
    def create_ai_model(self, name: str, provider: str, model_name: str, config: Dict[str, Any], is_active: bool = False) -> int:
        """Create AI model record"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            config_json = json.dumps(config) if config else "{}"
            
            cur.execute(
                f"INSERT INTO ai_models (name, provider, model_name, is_active, config) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (name, provider, model_name, is_active, config_json)
            )
            conn.commit()
            
            cur.execute(f"SELECT id FROM ai_models WHERE name = {placeholder}", (name,))
            model = cur.fetchone()
            return model[0] if model else None
        finally:
            conn.close()
    
    def get_all_ai_models(self) -> List[AIModel]:
        """Get all AI models"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT id, name, provider, model_name, is_active, config, created_at FROM ai_models ORDER BY created_at DESC")
            rows = cur.fetchall()
            
            models = []
            for row in rows:
                config = {}
                if row[5]:
                    try:
                        config = json.loads(row[5])
                    except:
                        config = {}
                
                models.append(AIModel(
                    id=row[0], name=row[1], provider=row[2], model_name=row[3],
                    is_active=bool(row[4]), config=config, created_at=row[6]
                ))
            return models
        finally:
            conn.close()
    
    def activate_ai_model(self, model_id: int) -> bool:
        """Activate an AI model and deactivate others"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"UPDATE ai_models SET is_active = FALSE")
            cur.execute(f"UPDATE ai_models SET is_active = TRUE WHERE id = {placeholder}", (model_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def get_active_ai_model(self) -> Optional[AIModel]:
        """Get the active AI model"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT id, name, provider, model_name, is_active, config, created_at FROM ai_models WHERE is_active = TRUE LIMIT 1")
            row = cur.fetchone()
            
            if row:
                config = {}
                if row[5]:
                    try:
                        config = json.loads(row[5])
                    except:
                        config = {}
                
                return AIModel(
                    id=row[0], name=row[1], provider=row[2], model_name=row[3],
                    is_active=bool(row[4]), config=config, created_at=row[6]
                )
            return None
        finally:
            conn.close()

    def delete_ai_model(self, model_id: int) -> bool:
        """Delete an AI model configuration"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            # First check if the model exists and is active
            cur.execute(f"SELECT is_active FROM ai_models WHERE id = {placeholder}", (model_id,))
            result = cur.fetchone()
            
            if not result:
                return False  # Model not found
                
            if result[0]:  # Check if is_active is True
                # Don't allow deletion of active model
                raise Exception("Cannot delete the active AI model. Please activate another model first.")
            
            # Delete the model
            cur.execute(f"DELETE FROM ai_models WHERE id = {placeholder}", (model_id,))
            conn.commit()
            return cur.rowcount > 0
            
        except Exception as e:
            conn.rollback()
            if "Cannot delete the active AI model" in str(e):
                raise e  # Re-raise the business logic error
            raise Exception(f"Database error while deleting AI model: {str(e)}")
        finally:
            conn.close()

    # Recently viewed projects methods
    def add_recently_viewed_project(self, user_id: int, project_id: str) -> bool:
        """Add or update recently viewed project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.use_rds:
                # PostgreSQL: use ON CONFLICT on the unique (user_id, project_id) constraint
                cur.execute(f"""
                    INSERT INTO recently_viewed_projects (user_id, project_id, view_count, viewed_at)
                    VALUES ({placeholder}, {placeholder}, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, project_id) DO UPDATE
                    SET view_count = recently_viewed_projects.view_count + 1,
                        viewed_at = CURRENT_TIMESTAMP
                """, (user_id, project_id))
            else:
                # SQLite: use INSERT OR REPLACE and a subselect to increment view_count
                cur.execute(f"""
                    INSERT OR REPLACE INTO recently_viewed_projects (user_id, project_id, view_count, viewed_at) 
                    VALUES ({placeholder}, {placeholder}, 
                    COALESCE((SELECT view_count FROM recently_viewed_projects WHERE user_id = {placeholder} AND project_id = {placeholder}), 0) + 1,
                    CURRENT_TIMESTAMP)
                """, (user_id, project_id, user_id, project_id))

            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
    
    def get_recently_viewed_projects(self, user_id: int, limit: int = 10) -> List[RecentlyViewedProject]:
        """Get user's recently viewed projects"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                SELECT rvp.id, rvp.user_id, rvp.project_id, rvp.viewed_at, rvp.view_count, p.name as project_name
                FROM recently_viewed_projects rvp
                JOIN projects p ON rvp.project_id = p.project_id
                WHERE rvp.user_id = {placeholder}
                ORDER BY rvp.viewed_at DESC
                LIMIT {placeholder}
            """, (user_id, limit))
            
            rows = cur.fetchall()
            
            return [
                RecentlyViewedProject(
                    id=row[0], user_id=row[1], project_id=row[2], viewed_at=row[3],
                    view_count=row[4], project_name=row[5] if len(row) > 5 else ""
                )
                for row in rows
            ]
        finally:
            conn.close()

    # User subscription methods
    def create_user_subscription(
        self,
        user_id: int,
        plan_id: int,
        stripe_subscription_id: str = None,
        stripe_customer_id: str = None,
        interval: str = SubscriptionInterval.SIX_MONTH.value,
        current_period_start: Optional[datetime] = None,
        current_period_end: Optional[datetime] = None,
        status: str = "active",
        auto_renew: bool = True,
        is_active: bool = True,
    ) -> int:
        """Create user subscription"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        interval_column = self._quote_identifier("interval")

        try:
            # FIXED: Use backticks around interval in the query
            cur.execute(f"""
                INSERT INTO user_subscriptions (
                    user_id, plan_id, stripe_subscription_id, stripe_customer_id, {interval_column},
                    current_period_start, current_period_end, status, auto_renew, is_active
                )
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (
                user_id,
                plan_id,
                stripe_subscription_id,
                stripe_customer_id,
                interval,
                current_period_start,
                current_period_end,
                status,
                auto_renew,
                is_active,
            ))
            conn.commit()

            cur.execute(f"SELECT id FROM user_subscriptions WHERE user_id = {placeholder} ORDER BY created_at DESC LIMIT 1", (user_id,))
            subscription = cur.fetchone()
            return subscription[0] if subscription else None
        finally:
            conn.close()
    
    def get_user_subscription(self, user_id: int) -> Optional[UserSubscription]:
        """Get user's active subscription"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        interval_column = self._quote_identifier("interval")
        active_flag = True if self.use_rds else 1

        try:
            # FIXED: Use backticks around interval in the query
            cur.execute(f"""
                SELECT us.id, us.user_id, us.plan_id, us.stripe_subscription_id, us.stripe_customer_id,
                    us.current_period_start, us.current_period_end, us.status, us.{interval_column} AS interval_value,
                    us.auto_renew, us.is_active, us.created_at, us.updated_at
                FROM user_subscriptions us
                WHERE us.user_id = {placeholder} AND us.is_active = {placeholder}
                ORDER BY us.created_at DESC
                LIMIT 1
            """, (user_id, active_flag))

            row = cur.fetchone()
            if row:
                return UserSubscription(
                    id=row[0], user_id=row[1], plan_id=row[2], stripe_subscription_id=row[3],
                    stripe_customer_id=row[4], current_period_start=row[5], current_period_end=row[6],
                    status=row[7], interval=row[8], auto_renew=bool(row[9]), is_active=bool(row[10]),
                    created_at=row[11], updated_at=row[12]
                )
            return None
        finally:
            conn.close()
    
    def get_user_subscription_by_stripe_id(self, stripe_subscription_id: str) -> Optional[UserSubscription]:
        """Get user subscription by Stripe ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        interval_column = self._quote_identifier("interval")

        try:
            # FIXED: Use backticks around interval in the query
            cur.execute(f"""
                SELECT id, user_id, plan_id, stripe_subscription_id, stripe_customer_id,
                       current_period_start, current_period_end, status, {interval_column} AS interval_value,
                       auto_renew, is_active, created_at, updated_at
                FROM user_subscriptions
                WHERE stripe_subscription_id = {placeholder}
            """, (stripe_subscription_id,))

            row = cur.fetchone()
            if row:
                return UserSubscription(
                    id=row[0], user_id=row[1], plan_id=row[2], stripe_subscription_id=row[3],
                    stripe_customer_id=row[4], current_period_start=row[5], current_period_end=row[6],
                    status=row[7], interval=row[8], auto_renew=bool(row[9]), is_active=bool(row[10]),
                    created_at=row[11], updated_at=row[12]
                )
            return None
        finally:
            conn.close()
    
    def update_user_subscription(self, subscription_id: int, **kwargs) -> bool:
        """Update user subscription"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        interval_column = self._quote_identifier("interval")

        allowed_columns = {
            'plan_id': 'plan_id',
            'stripe_subscription_id': 'stripe_subscription_id',
            'stripe_customer_id': 'stripe_customer_id',
            'current_period_start': 'current_period_start',
            'current_period_end': 'current_period_end',
            'status': 'status',
            'interval': interval_column,
            'auto_renew': 'auto_renew',
            'is_active': 'is_active',
        }

        try:
            update_fields = []
            params = []

            for key, value in kwargs.items():
                if value is None or key not in allowed_columns:
                    continue

                column_name = allowed_columns[key]
                update_fields.append(f"{column_name} = {placeholder}")
                params.append(value)

            if not update_fields:
                return False

            if self.use_rds:
                update_fields.append("updated_at = CURRENT_TIMESTAMP")
            else:
                update_fields.append("updated_at = datetime('now')")

            params.append(subscription_id)
            query = f"UPDATE user_subscriptions SET {', '.join(update_fields)} WHERE id = {placeholder}"
            cur.execute(query, params)
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # Dashboard statistics methods
    def get_total_users_count(self, status: str = None, search: str = None) -> int:
        """Get total users count with optional filtering"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            query = "SELECT COUNT(*) FROM userdata WHERE 1=1"
            params = []
            
            if status:
                query += f" AND is_active = {placeholder}"
                params.append(status == "active")
            
            if search:
                query += f" AND (email LIKE {placeholder} OR firstname LIKE {placeholder} OR lastname LIKE {placeholder})"
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term])
            
            cur.execute(query, params)
            return cur.fetchone()[0]
        finally:
            conn.close()
    
    def get_active_users_count(self) -> int:
        """Get count of active users"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT COUNT(*) FROM userdata WHERE is_active = TRUE")
            return cur.fetchone()[0]
        finally:
            conn.close()
    
    def get_total_feedback_count(self, rating: str = None) -> int:
        """Get total feedback count with optional rating filter"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            query = "SELECT COUNT(*) FROM feedback"
            params = []
            
            if rating:
                query += f" WHERE rating = {placeholder}"
                params.append(rating)
            
            cur.execute(query, params)
            return cur.fetchone()[0]
        finally:
            conn.close()
    
    def get_recent_signups(self, days: int = 7) -> int:
        """Get recent signups count"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            # Validate input to prevent injection
            if not isinstance(days, int) or days <= 0:
                days = 7
                
            if self.use_rds:
                # For MySQL: Use string formatting with validation
                query = f"SELECT COUNT(*) FROM userdata WHERE created_at >= DATE_SUB(NOW(), INTERVAL {days} DAY)"
                cur.execute(query)
            else:
                # For SQLite: Use parameterized query
                cur.execute("SELECT COUNT(*) FROM userdata WHERE created_at >= datetime('now', '-? days')", (days,))
            
            return cur.fetchone()[0]
        finally:
            conn.close()


    def get_total_storage_usage(self) -> int:
        """Get total storage usage in MB"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT SUM(used_storage_mb) FROM user_storage")
            result = cur.fetchone()[0]
            return result if result else 0
        finally:
            conn.close()

    def get_dashboard_metrics(self, days: int = 14) -> Dict[str, Any]:
        """Get time series metrics for the admin dashboard graph."""

        if days < 1:
            days = 14

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days - 1)
        labels = [start_date + timedelta(days=i) for i in range(days)]

        def _as_date(value) -> Optional[datetime.date]:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, str):
                for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                    try:
                        return datetime.strptime(value[:26], fmt).date()
                    except ValueError:
                        continue
                try:
                    return datetime.fromisoformat(value).date()
                except Exception:
                    return None
            return None

        datasets = {
            "new_users": ("userdata", "created_at"),
            "new_projects": ("projects", "created_at"),
            "new_subscriptions": ("user_subscriptions", "created_at"),
        }

        placeholder = self._get_placeholder()
        metrics = {key: {label: 0 for label in labels} for key in datasets.keys()}

        conn = self.get_connection()
        cur = conn.cursor()

        try:
            for key, (table, column) in datasets.items():
                query = (
                    f"SELECT {column} FROM {table} "
                    f"WHERE {column} >= {placeholder} AND {column} < {placeholder}"
                )
                cur.execute(query, (start_date, end_date + timedelta(days=1)))
                rows = cur.fetchall()
                for row in rows:
                    date_value = _as_date(row[0])
                    if date_value and start_date <= date_value <= end_date:
                        metrics[key][date_value] += 1

            labels_iso = [label.isoformat() for label in labels]
            series = {
                key: [metrics[key][label] for label in labels]
                for key in metrics
            }
            totals = {key: sum(values) for key, values in series.items()}

            return {
                "labels": labels_iso,
                "series": series,
                "totals": totals,
                "range": {
                    "start": labels_iso[0] if labels_iso else None,
                    "end": labels_iso[-1] if labels_iso else None,
                },
            }
        finally:
            conn.close()
    
    def get_subscriptions_expiring_soon(self, days: int) -> List[Dict]:
        """Get subscriptions expiring soon"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            if self.use_rds:
                # PostgreSQL syntax - remove DATE_ADD and use INTERVAL directly
                query = """
                    SELECT us.id, u.email, sp.name, us.current_period_end 
                    FROM user_subscriptions us
                    JOIN userdata u ON us.user_id = u.id
                    JOIN subscription_plans sp ON us.plan_id = sp.id
                    WHERE us.is_active = TRUE 
                    AND us.current_period_end BETWEEN NOW() AND (NOW() + INTERVAL '%s DAY')
                """
                cur.execute(query, (days,))
            else:
                # SQLite syntax (unchanged)
                cur.execute(f"""
                    SELECT us.id, u.email, sp.name, us.current_period_end 
                    FROM user_subscriptions us
                    JOIN userdata u ON us.user_id = u.id
                    JOIN subscription_plans sp ON us.plan_id = sp.id
                    WHERE us.is_active = TRUE 
                    AND us.current_period_end BETWEEN datetime('now') AND datetime('now', '+? days')
                """, (days,))
            
            rows = cur.fetchall()
            return [
                {
                    "subscription_id": row[0],
                    "user_email": row[1],
                    "plan_name": row[2],
                    "expiry_date": row[3]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_recently_expired_subscriptions(self, days: int) -> List[Dict]:
        """Get recently expired subscriptions"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            if self.use_rds:
                # PostgreSQL syntax - remove DATE_SUB
                query = """
                    SELECT us.id, u.email, sp.name, us.current_period_end 
                    FROM user_subscriptions us
                    JOIN userdata u ON us.user_id = u.id
                    JOIN subscription_plans sp ON us.plan_id = sp.id
                    WHERE us.is_active = TRUE 
                    AND us.current_period_end BETWEEN (NOW() - INTERVAL '%s DAY') AND NOW()
                """
                cur.execute(query, (days,))
            else:
                # SQLite syntax (unchanged)
                cur.execute(f"""
                    SELECT us.id, u.email, sp.name, us.current_period_end 
                    FROM user_subscriptions us
                    JOIN userdata u ON us.user_id = u.id
                    JOIN subscription_plans sp ON us.plan_id = sp.id
                    WHERE us.is_active = TRUE 
                    AND us.current_period_end BETWEEN datetime('now', '-? days') AND datetime('now')
                """, (days,))
            
            rows = cur.fetchall()
            return [
                {
                    "subscription_id": row[0],
                    "user_email": row[1],
                    "plan_name": row[2],
                    "expiry_date": row[3]
                }
                for row in rows
            ]
        finally:
            conn.close()
    # Email verification methods
    def create_verification_token(
        self,
        user_id: int,
        token: str,
        expires_at: datetime,
        purpose: str = "email_verification"
    ) -> bool:
        """Create or update verification/OTP token for user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            # Update legacy verification columns only for email verification purpose
            if purpose == "email_verification":
                cur.execute(
                    f"UPDATE userdata SET verification_token = {placeholder}, verification_token_expires = {placeholder} WHERE id = {placeholder}",
                    (token, expires_at, user_id)
                )

            # Ensure only one active OTP per user/purpose
            cur.execute(
                f"DELETE FROM user_otp_codes WHERE user_id = {placeholder} AND purpose = {placeholder}",
                (user_id, purpose)
            )

            cur.execute(
                f"INSERT INTO user_otp_codes (user_id, otp_code, purpose, expires_at) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (user_id, token, purpose, expires_at)
            )

            conn.commit()
            return True
        finally:
            conn.close()
    
    def get_user_by_verification_token(self, token: str) -> Optional[User]:
        """Get user by verification token"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, firstname, lastname, email, password, google_id, is_verified, verification_token, verification_token_expires, created_at FROM userdata WHERE verification_token = {placeholder}",
                (token,)
            )
            row = cur.fetchone()
            
            if row:
                return User(
                    id=row[0], firstname=row[1], lastname=row[2], email=row[3], password=row[4],
                    google_id=row[5], is_verified=bool(row[6]), verification_token=row[7],
                    verification_token_expires=row[8], created_at=row[9]
                )
            return None
        finally:
            conn.close()

    def get_user_otp(self, user_id: int, purpose: str) -> Optional[UserOTP]:
        """Retrieve the most recent active OTP for a user and purpose"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, user_id, otp_code, purpose, expires_at, created_at, consumed_at FROM user_otp_codes WHERE user_id = {placeholder} AND purpose = {placeholder} ORDER BY created_at DESC LIMIT 1",
                (user_id, purpose)
            )
            row = cur.fetchone()

            if row:
                return UserOTP(
                    id=row[0],
                    user_id=row[1],
                    otp_code=row[2],
                    purpose=row[3],
                    expires_at=row[4],
                    created_at=row[5],
                    consumed_at=row[6]
                )
            return None
        finally:
            conn.close()

    def get_user_otp_by_code(self, user_id: int, otp_code: str) -> Optional[UserOTP]:
        """Retrieve the most recent active OTP for a user by code regardless of purpose"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, user_id, otp_code, purpose, expires_at, created_at, consumed_at FROM user_otp_codes WHERE user_id = {placeholder} AND otp_code = {placeholder} ORDER BY created_at DESC LIMIT 1",
                (user_id, otp_code)
            )
            row = cur.fetchone()

            if row:
                return UserOTP(
                    id=row[0],
                    user_id=row[1],
                    otp_code=row[2],
                    purpose=row[3],
                    expires_at=row[4],
                    created_at=row[5],
                    consumed_at=row[6]
                )
            return None
        finally:
            conn.close()

    def consume_user_otp(self, user_id: int, otp_code: str, purpose: str) -> bool:
        """Mark an OTP as consumed"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"UPDATE user_otp_codes SET consumed_at = CURRENT_TIMESTAMP WHERE user_id = {placeholder} AND otp_code = {placeholder} AND purpose = {placeholder} AND consumed_at IS NULL",
                (user_id, otp_code, purpose)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def verify_user_email(self, user_id: int) -> bool:
        """Mark user email as verified and clear verification token"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"UPDATE userdata SET is_verified = {1 if not self.use_rds else 'TRUE'}, verification_token = NULL, verification_token_expires = NULL WHERE id = {placeholder}",
                (user_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def clear_expired_verification_tokens(self):
        """Clear expired verification tokens"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            if self.use_rds:
                cur.execute(
                    "UPDATE userdata SET verification_token = NULL, verification_token_expires = NULL WHERE verification_token_expires < NOW()"
                )
            else:
                cur.execute(
                    "UPDATE userdata SET verification_token = NULL, verification_token_expires = NULL WHERE verification_token_expires < datetime('now')"
                )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def clear_expired_otps(self):
        """Remove expired OTP codes"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            if self.use_rds:
                cur.execute("DELETE FROM user_otp_codes WHERE expires_at < NOW() OR consumed_at IS NOT NULL")
            else:
                cur.execute("DELETE FROM user_otp_codes WHERE expires_at < datetime('now') OR consumed_at IS NOT NULL")
            conn.commit()
        finally:
            conn.close()

    # Chat and session management methods
    def add_chat_message(self, user_id: int, session_id: str, role: str, message: str, context_type: str = None, context_id: str = None):
        """Add a chat message to history with optional context information"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                INSERT INTO chathistory (user_id, session_id, role, message, context_type, context_id) 
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (user_id, session_id, role, message, context_type, context_id))
            
            conn.commit()
            
            # Update session activity if session exists
            self.update_session_activity(session_id)
        finally:
            conn.close()
    
    def get_chat_history(self, user_id: int, session_id: str = None, limit: int = 50) -> List[ChatMessage]:
        """Get chat history for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if session_id:
                if self.use_rds:
                    cur.execute("""
                        SELECT id, user_id, session_id, role, message, timestamp, context_type, context_id
                        FROM chathistory 
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (user_id, session_id, limit))
                else:
                    cur.execute("""
                        SELECT id, user_id, session_id, role, message, timestamp, context_type, context_id
                        FROM chathistory 
                        WHERE user_id = ? AND session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (user_id, session_id, limit))
            else:
                if self.use_rds:
                    cur.execute("""
                        SELECT id, user_id, session_id, role, message, timestamp, context_type, context_id
                        FROM chathistory 
                        WHERE user_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (user_id, limit))
                else:
                    cur.execute("""
                        SELECT id, user_id, session_id, role, message, timestamp, context_type, context_id
                        FROM chathistory 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """, (user_id, limit))
            
            rows = cur.fetchall()
            return [
                ChatMessage(
                    id=row[0],
                    user_id=row[1],
                    session_id=row[2],
                    role=row[3],
                    message=row[4],
                    timestamp=row[5],
                    context_type=row[6],
                    context_id=row[7]
                )
                for row in rows
            ]
            
        finally:
            conn.close()
    
    def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all unique session IDs for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                SELECT DISTINCT session_id, MAX(timestamp) as last_activity
                FROM chathistory 
                WHERE user_id = {placeholder}
                GROUP BY session_id
                ORDER BY last_activity DESC
            """, (user_id,))
            
            sessions = cur.fetchall()
            return [
                {
                    "session_id": session[0],
                    "last_activity": session[1]
                }
                for session in sessions
            ]
            
        finally:
            conn.close()
    
    def get_project_session(self, user_id: int, project_id: str) -> Optional[str]:
        """Get the most recent session ID associated with a specific project for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            # Search for chat messages that mention the project ID
            # This looks for messages that contain the project ID in the content
            cur.execute(f"""
                SELECT session_id, MAX(timestamp) as last_activity
                FROM chathistory 
                WHERE user_id = {placeholder} 
                AND (message LIKE {placeholder} OR message LIKE {placeholder})
                GROUP BY session_id
                ORDER BY last_activity DESC
                LIMIT 1
            """, (user_id, f"%Project ID: {project_id}%", f"%project_id%{project_id}%"))
            
            row = cur.fetchone()
            return row[0] if row else None
            
        except Exception as e:
            # If there's an error, return None to fall back to creating a new session
            return None
        finally:
            conn.close()
    
    # Enhanced session management methods
    def create_chat_session(self, session_id: str, user_id: int, context_type: str, context_id: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Create a new chat session with context support"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            cur.execute(f"""
                INSERT INTO chat_sessions (session_id, user_id, context_type, context_id, metadata)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (session_id, user_id, context_type, context_id, metadata_json))
            
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            print(f"Error creating chat session: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        """Get session by session ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                SELECT id, session_id, user_id, context_type, context_id, is_active, 
                       created_at, last_activity, metadata
                FROM chat_sessions 
                WHERE session_id = {placeholder}
            """, (session_id,))
            
            row = cur.fetchone()
            if row:
                metadata = json.loads(row[8]) if row[8] else None
                return ChatSession(
                    id=row[0],
                    session_id=row[1],
                    user_id=row[2],
                    context_type=row[3],
                    context_id=row[4],
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_activity=row[7],
                    metadata=metadata
                )
            return None
        finally:
            conn.close()
    
    def get_session_by_context(self, user_id: int, context_type: str, context_id: str = None) -> Optional[ChatSession]:
        """Get the most recent active session for a specific context"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if context_id is None:
                cur.execute(f"""
                    SELECT id, session_id, user_id, context_type, context_id, is_active, 
                           created_at, last_activity, metadata
                    FROM chat_sessions 
                    WHERE user_id = {placeholder} AND context_type = {placeholder} AND context_id IS NULL
                    AND is_active = {'TRUE' if self.use_rds else '1'}
                    ORDER BY last_activity DESC
                    LIMIT 1
                """, (user_id, context_type))
            else:
                cur.execute(f"""
                    SELECT id, session_id, user_id, context_type, context_id, is_active, 
                           created_at, last_activity, metadata
                    FROM chat_sessions 
                    WHERE user_id = {placeholder} AND context_type = {placeholder} AND context_id = {placeholder}
                    AND is_active = {'TRUE' if self.use_rds else '1'}
                    ORDER BY last_activity DESC
                    LIMIT 1
                """, (user_id, context_type, context_id))
            
            row = cur.fetchone()
            if row:
                metadata = json.loads(row[8]) if row[8] else None
                return ChatSession(
                    id=row[0],
                    session_id=row[1],
                    user_id=row[2],
                    context_type=row[3],
                    context_id=row[4],
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_activity=row[7],
                    metadata=metadata
                )
            return None
        finally:
            conn.close()
    
    def get_active_sessions(self, user_id: int, context_type: str = None) -> List[ChatSession]:
        """Get all active sessions for a user, optionally filtered by context type"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if context_type:
                cur.execute(f"""
                    SELECT id, session_id, user_id, context_type, context_id, is_active, 
                           created_at, last_activity, metadata
                    FROM chat_sessions 
                    WHERE user_id = {placeholder} AND context_type = {placeholder}
                    AND is_active = {'TRUE' if self.use_rds else '1'}
                    ORDER BY last_activity DESC
                """, (user_id, context_type))
            else:
                cur.execute(f"""
                    SELECT id, session_id, user_id, context_type, context_id, is_active, 
                           created_at, last_activity, metadata
                    FROM chat_sessions 
                    WHERE user_id = {placeholder}
                    AND is_active = {'TRUE' if self.use_rds else '1'}
                    ORDER BY last_activity DESC
                """, (user_id,))
            
            rows = cur.fetchall()
            sessions = []
            for row in rows:
                metadata = json.loads(row[8]) if row[8] else None
                sessions.append(ChatSession(
                    id=row[0],
                    session_id=row[1],
                    user_id=row[2],
                    context_type=row[3],
                    context_id=row[4],
                    is_active=bool(row[5]),
                    created_at=row[6],
                    last_activity=row[7],
                    metadata=metadata
                ))
            return sessions
        finally:
            conn.close()
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update the last activity timestamp for a session"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.use_rds:
                # MySQL automatically updates last_activity with ON UPDATE CURRENT_TIMESTAMP
                cur.execute(f"""
                    UPDATE chat_sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_id = {placeholder}
                """, (session_id,))
            else:
                # SQLite needs manual timestamp update
                cur.execute(f"""
                    UPDATE chat_sessions 
                    SET last_activity = CURRENT_TIMESTAMP 
                    WHERE session_id = {placeholder}
                """, (session_id,))
            
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            print(f"Error updating session activity: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def deactivate_session(self, session_id: str) -> bool:
        """Mark a session as inactive"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                UPDATE chat_sessions 
                SET is_active = {'FALSE' if self.use_rds else '0'}
                WHERE session_id = {placeholder}
            """, (session_id,))
            
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            print(f"Error deactivating session: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def cleanup_expired_sessions(self, hours: int = 24) -> int:
        """Mark sessions as inactive if they haven't been active for the specified hours"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.use_rds:
                cur.execute(f"""
                    UPDATE chat_sessions 
                    SET is_active = FALSE
                    WHERE is_active = TRUE 
                    AND last_activity < NOW() - INTERVAL '{placeholder} hours'
                """, (hours,))
            else:
                cur.execute(f"""
                    UPDATE chat_sessions 
                    SET is_active = 0
                    WHERE is_active = 1 
                    AND datetime(last_activity, '+{placeholder} hours') < datetime('now')
                """, (hours,))
            
            conn.commit()
            cleaned_count = cur.rowcount
            print(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
        except Exception as e:
            print(f"Error cleaning up expired sessions: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def add_chat_message_with_context(self, user_id: int, session_id: str, role: str, message: str, context_type: str = None, context_id: str = None):
        """Add a chat message to history with context information"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                INSERT INTO chathistory (user_id, session_id, role, message, context_type, context_id) 
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """, (user_id, session_id, role, message, context_type, context_id))
            
            conn.commit()
            
            # Update session activity
            self.update_session_activity(session_id)
        finally:
            conn.close()
    
    # Project management methods
    def create_project(self, project_id: str, name: str, description: str, user_id: int, doc_ids: Optional[List[str]] = None) -> int:
        """Create a new project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            # Convert doc_ids list to JSON string if present
            doc_ids_str = None
            if doc_ids and len(doc_ids) > 0:
                doc_ids_str = json.dumps(doc_ids)
            
            cur.execute(
                f"INSERT INTO projects (project_id, name, description, user_id, doc_ids) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (project_id, name, description, user_id, doc_ids_str)
            )
            conn.commit()
            
            cur.execute(f"SELECT id FROM projects WHERE project_id = {placeholder}", (project_id,))
            project = cur.fetchone()
            return project[0] if project else None
        finally:
            conn.close()
    
    def get_user_projects(self, user_id: int) -> List[Project]:
        """Get all projects for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, project_id, name, description, user_id, doc_ids, created_at, updated_at FROM projects WHERE user_id = {placeholder} ORDER BY created_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()
            
            result = []
            for row in rows:
                doc_ids = None
                if row[5]:
                    try:
                        doc_ids = json.loads(row[5])
                    except:
                        doc_ids = [row[5]]
                
                result.append(Project(
                    id=row[0], project_id=row[1], name=row[2], description=row[3],
                    user_id=row[4], doc_ids=doc_ids, created_at=row[6], updated_at=row[7]
                ))
            
            return result
        finally:
            conn.close()

    def get_shared_projects_for_user(self, user_id: int) -> List[Dict[str, Any]]:
        """Get projects shared with the user along with membership role"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT p.id, p.project_id, p.name, p.description, p.user_id, p.doc_ids, p.created_at, p.updated_at, pm.role FROM projects p INNER JOIN project_members pm ON p.project_id = pm.project_id WHERE pm.user_id = {placeholder} ORDER BY pm.created_at DESC",
                (user_id,)
            )
            rows = cur.fetchall()

            shared = []
            for row in rows:
                doc_ids = None
                if row[5]:
                    try:
                        doc_ids = json.loads(row[5])
                    except Exception:
                        doc_ids = [row[5]]

                shared.append({
                    "project": Project(
                        id=row[0],
                        project_id=row[1],
                        name=row[2],
                        description=row[3],
                        user_id=row[4],
                        doc_ids=doc_ids,
                        created_at=row[6],
                        updated_at=row[7]
                    ),
                    "role": row[8]
                })
            return shared
        finally:
            conn.close()

    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get a single project by its project_id"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"SELECT id, project_id, name, description, user_id, doc_ids, created_at, updated_at FROM projects WHERE project_id = {placeholder}",
                (project_id,)
            )
            row = cur.fetchone()
            
            if not row:
                return None
            
            doc_ids = None
            if row[5]:
                try:
                    doc_ids = json.loads(row[5])
                except:
                    doc_ids = [row[5]]
            
            return Project(
                id=row[0], project_id=row[1], name=row[2], description=row[3],
                user_id=row[4], doc_ids=doc_ids, created_at=row[6], updated_at=row[7]
            )
        finally:
            conn.close()

    def get_project_members(self, project_id: str) -> List[ProjectMember]:
        """Retrieve members for a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, project_id, user_id, role, created_at FROM project_members WHERE project_id = {placeholder}",
                (project_id,)
            )
            rows = cur.fetchall()
            return [
                ProjectMember(id=row[0], project_id=row[1], user_id=row[2], role=row[3], created_at=row[4])
                for row in rows
            ]
        finally:
            conn.close()

    def add_project_member(self, project_id: str, user_id: int, role: str = "member") -> bool:
        """Add a member to a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id FROM project_members WHERE project_id = {placeholder} AND user_id = {placeholder}",
                (project_id, user_id)
            )
            if cur.fetchone():
                return False

            cur.execute(
                f"INSERT INTO project_members (project_id, user_id, role) VALUES ({placeholder}, {placeholder}, {placeholder})",
                (project_id, user_id, role)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def remove_project_member(self, project_id: str, user_id: int) -> bool:
        """Remove a member from a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"DELETE FROM project_members WHERE project_id = {placeholder} AND user_id = {placeholder}",
                (project_id, user_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def user_has_project_access(self, project_id: str, user_id: int) -> bool:
        """Check if user is owner or member of project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT 1 FROM projects WHERE project_id = {placeholder} AND user_id = {placeholder}",
                (project_id, user_id)
            )
            if cur.fetchone():
                return True

            cur.execute(
                f"SELECT 1 FROM project_members WHERE project_id = {placeholder} AND user_id = {placeholder}",
                (project_id, user_id)
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def get_pending_project_invitation(self, project_id: str, invitee_id: int) -> Optional[ProjectInvitation]:
        """Fetch a pending invitation for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, project_id, inviter_id, invitee_id, invitee_email, role, status, created_at, responded_at FROM project_invitations WHERE project_id = {placeholder} AND invitee_id = {placeholder} AND status = 'pending'",
                (project_id, invitee_id)
            )
            row = cur.fetchone()

            if row:
                return ProjectInvitation(
                    id=row[0],
                    project_id=row[1],
                    inviter_id=row[2],
                    invitee_id=row[3],
                    invitee_email=row[4],
                    role=row[5],
                    status=row[6],
                    created_at=row[7],
                    responded_at=row[8]
                )
            return None
        finally:
            conn.close()

    def create_project_invitation(
        self,
        project_id: str,
        inviter_id: int,
        invitee_id: int,
        invitee_email: str,
        role: str = "member"
    ) -> int:
        """Create a new project invitation"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            # Remove existing pending invitation for same user
            cur.execute(
                f"DELETE FROM project_invitations WHERE project_id = {placeholder} AND invitee_id = {placeholder} AND status = 'pending'",
                (project_id, invitee_id)
            )

            cur.execute(
                f"INSERT INTO project_invitations (project_id, inviter_id, invitee_id, invitee_email, role) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (project_id, inviter_id, invitee_id, invitee_email, role)
            )
            conn.commit()

            cur.execute(f"SELECT id FROM project_invitations WHERE project_id = {placeholder} AND invitee_id = {placeholder} ORDER BY created_at DESC LIMIT 1", (project_id, invitee_id))
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_project_invitation_by_id(self, invitation_id: int) -> Optional[ProjectInvitation]:
        """Retrieve invitation by ID"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"SELECT id, project_id, inviter_id, invitee_id, invitee_email, role, status, created_at, responded_at FROM project_invitations WHERE id = {placeholder}",
                (invitation_id,)
            )
            row = cur.fetchone()

            if row:
                return ProjectInvitation(
                    id=row[0],
                    project_id=row[1],
                    inviter_id=row[2],
                    invitee_id=row[3],
                    invitee_email=row[4],
                    role=row[5],
                    status=row[6],
                    created_at=row[7],
                    responded_at=row[8]
                )
            return None
        finally:
            conn.close()

    def update_project_invitation_status(self, invitation_id: int, status: str) -> bool:
        """Update invitation status"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            cur.execute(
                f"UPDATE project_invitations SET status = {placeholder}, responded_at = CURRENT_TIMESTAMP WHERE id = {placeholder}",
                (status, invitation_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def get_user_project_invitations(self, user_id: int, status: Optional[str] = None) -> List[ProjectInvitation]:
        """List invitations for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            if status:
                cur.execute(
                    f"SELECT id, project_id, inviter_id, invitee_id, invitee_email, role, status, created_at, responded_at FROM project_invitations WHERE invitee_id = {placeholder} AND status = {placeholder} ORDER BY created_at DESC",
                    (user_id, status)
                )
            else:
                cur.execute(
                    f"SELECT id, project_id, inviter_id, invitee_id, invitee_email, role, status, created_at, responded_at FROM project_invitations WHERE invitee_id = {placeholder} ORDER BY created_at DESC",
                    (user_id,)
                )

            rows = cur.fetchall()
            return [
                ProjectInvitation(
                    id=row[0],
                    project_id=row[1],
                    inviter_id=row[2],
                    invitee_id=row[3],
                    invitee_email=row[4],
                    role=row[5],
                    status=row[6],
                    created_at=row[7],
                    responded_at=row[8]
                )
                for row in rows
            ]
        finally:
            conn.close()

    def create_notification(self, user_id: int, title: str, message: str, notification_type: str = "general", metadata: Dict[str, Any] = None) -> bool:
        """Create in-app notification"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        metadata_json = json.dumps(metadata) if metadata else None

        try:
            cur.execute(
                f"INSERT INTO notifications (user_id, title, message, notification_type, metadata) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                (user_id, title, message, notification_type, metadata_json)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def get_notifications(self, user_id: int, include_read: bool = False, limit: int = 50) -> List[Notification]:
        """Retrieve notifications for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            if include_read:
                cur.execute(
                    f"SELECT id, user_id, title, message, notification_type, metadata, is_read, created_at FROM notifications WHERE user_id = {placeholder} ORDER BY created_at DESC LIMIT {limit}",
                    (user_id,)
                )
            else:
                cur.execute(
                    f"SELECT id, user_id, title, message, notification_type, metadata, is_read, created_at FROM notifications WHERE user_id = {placeholder} AND is_read = {0 if not self.use_rds else 'FALSE'} ORDER BY created_at DESC LIMIT {limit}",
                    (user_id,)
                )

            rows = cur.fetchall()
            notifications = []
            for row in rows:
                metadata_value = row[5]
                if metadata_value and isinstance(metadata_value, str):
                    try:
                        metadata_value = json.loads(metadata_value)
                    except json.JSONDecodeError:
                        metadata_value = {"raw": metadata_value}

                notifications.append(Notification(
                    id=row[0],
                    user_id=row[1],
                    title=row[2],
                    message=row[3],
                    notification_type=row[4],
                    metadata=metadata_value,
                    is_read=bool(row[6]),
                    created_at=row[7]
                ))
            return notifications
        finally:
            conn.close()

    def mark_notification_read(self, notification_id: int, user_id: int) -> bool:
        """Mark notification as read"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()

        try:
            value_true = 1 if not self.use_rds else 'TRUE'
            cur.execute(
                f"UPDATE notifications SET is_read = {value_true} WHERE id = {placeholder} AND user_id = {placeholder}",
                (notification_id, user_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # Document management methods
    def create_document(self, doc_id: str, filename: str, file_id: str, pages: int, chunks_indexed: int, user_id: int, pdf_path: str = None, vector_path: str = None) -> int:
        """Create a new document record"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.is_postgres:
                # Use new schema with file_id
                cur.execute(
                    f"INSERT INTO documents (doc_id, filename, file_id, pages, chunks_indexed, user_id) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                    (doc_id, filename, file_id, pages, chunks_indexed, user_id)
                )
            else:
                # Use old schema for backward compatibility
                cur.execute(
                    f"INSERT INTO documents (doc_id, filename, pdf_path, vector_path, pages, chunks_indexed, user_id) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})",
                    (doc_id, filename, pdf_path or "", vector_path or "", pages, chunks_indexed, user_id)
                )
            
            conn.commit()
            
            # Get the new document's ID
            cur.execute(f"SELECT id FROM documents WHERE doc_id = {placeholder}", (doc_id,))
            document = cur.fetchone()
            return document[0] if document else None
            
        finally:
            conn.close()
    
    def update_document_chunks_indexed(self, doc_id: str, chunks_indexed: int) -> bool:
        """Update the chunks_indexed count for a document"""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            placeholder = self._get_placeholder()
            
            cur.execute(
                f"UPDATE documents SET chunks_indexed = {placeholder} WHERE doc_id = {placeholder}",
                (chunks_indexed, doc_id)
            )
            conn.commit()
            
            return cur.rowcount > 0
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error updating document chunks count: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_document_by_doc_id(self, doc_id: str) -> Optional[Document]:
        """Get document by doc_id"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.is_postgres:
                # Use new schema with file_id
                cur.execute(f"""
                    SELECT id, doc_id, filename, file_id, pages, chunks_indexed, status, user_id, created_at, updated_at 
                    FROM documents 
                    WHERE doc_id = {placeholder}
                """, (doc_id,))
                row = cur.fetchone()
                
                if row:
                    return Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id=row[3],
                        pages=row[4],
                        chunks_indexed=row[5],
                        status=row[6],
                        user_id=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
            else:
                # Use old schema for backward compatibility (SQLite/MySQL)
                cur.execute(f"""
                    SELECT id, doc_id, filename, pages, chunks_indexed, status, user_id, created_at, updated_at 
                    FROM documents 
                    WHERE doc_id = {placeholder}
                """, (doc_id,))
                row = cur.fetchone()
                
                if row:
                    return Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id="",  # Not available in old schema
                        pages=row[3],
                        chunks_indexed=row[4],
                        status=row[5],
                        user_id=row[6],
                        created_at=row[7],
                        updated_at=row[8]
                    )
            return None
            
        finally:
            conn.close()
    
    def get_user_documents(self, user_id: int) -> List[Document]:
        """Get all documents for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.is_postgres:
                # Use new schema with file_id
                cur.execute(
                    f"SELECT id, doc_id, filename, file_id, pages, chunks_indexed, status, user_id, created_at, updated_at FROM documents WHERE user_id = {placeholder} ORDER BY created_at DESC",
                    (user_id,)
                )
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id=row[3],
                        pages=row[4],
                        chunks_indexed=row[5],
                        status=row[6],
                        user_id=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
                    for row in rows
                ]
            else:
                # Use old schema for backward compatibility
                cur.execute(
                    f"SELECT id, doc_id, filename, pages, chunks_indexed, status, user_id, created_at, updated_at FROM documents WHERE user_id = {placeholder} ORDER BY created_at DESC",
                    (user_id,)
                )
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id="",  # Not available in old schema
                        pages=row[3],
                        chunks_indexed=row[4],
                        status=row[5],
                        user_id=row[6],
                        created_at=row[7],
                        updated_at=row[8]
                    )
                    for row in rows
                ]
            
        finally:
            conn.close()
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the system"""
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            if self.is_postgres:
                # Use new schema with file_id
                cur.execute("""
                    SELECT id, doc_id, filename, file_id, pages, chunks_indexed, status, user_id, created_at, updated_at 
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id=row[3],
                        pages=row[4],
                        chunks_indexed=row[5],
                        status=row[6],
                        user_id=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
                    for row in rows
                ]
            else:
                # Use old schema for backward compatibility
                cur.execute("""
                    SELECT id, doc_id, filename, pages, chunks_indexed, status, user_id, created_at, updated_at 
                    FROM documents 
                    ORDER BY created_at DESC
                """)
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id="",  # Not available in old schema
                        pages=row[3],
                        chunks_indexed=row[4],
                        status=row[5],
                        user_id=row[6],
                        created_at=row[7],
                        updated_at=row[8]
                    )
                    for row in rows
                ]
            
        finally:
            conn.close()

    def get_project_documents(self, project_id: str) -> List[Document]:
        """Get all documents associated with a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.is_postgres:
                # Use new schema with file_id
                cur.execute(f"""
                    SELECT d.id, d.doc_id, d.filename, d.file_id, d.pages, d.chunks_indexed, d.status, d.user_id, d.created_at, d.updated_at 
                    FROM documents d
                    INNER JOIN project_documents pd ON d.doc_id = pd.doc_id
                    WHERE pd.project_id = {placeholder}
                    ORDER BY pd.created_at ASC
                """, (project_id,))
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id=row[3],
                        pages=row[4],
                        chunks_indexed=row[5],
                        status=row[6],
                        user_id=row[7],
                        created_at=row[8],
                        updated_at=row[9]
                    )
                    for row in rows
                ]
            else:
                # Use old schema for backward compatibility
                cur.execute(f"""
                    SELECT d.id, d.doc_id, d.filename, d.pages, d.chunks_indexed, d.status, d.user_id, d.created_at, d.updated_at 
                    FROM documents d
                    INNER JOIN project_documents pd ON d.doc_id = pd.doc_id
                    WHERE pd.project_id = {placeholder}
                    ORDER BY pd.created_at ASC
                """, (project_id,))
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row[0],
                        doc_id=row[1],
                        filename=row[2],
                        file_id="",  # Not available in old schema
                        pages=row[3],
                        chunks_indexed=row[4],
                        status=row[5],
                        user_id=row[6],
                        created_at=row[7],
                        updated_at=row[8]
                    )
                    for row in rows
                ]
            
        finally:
            conn.close()

    def update_project_document(self, project_id: str, doc_ids: List[str]) -> bool:
        """Update the document IDs for a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            # Convert doc_ids list to JSON string
            doc_ids_str = json.dumps(doc_ids)
            cur.execute(
                f"UPDATE projects SET doc_ids = {placeholder} WHERE project_id = {placeholder}",
                (doc_ids_str, project_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document record and any project-document links."""
        conn = None
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            placeholder = self._get_placeholder()

            # Remove any project-document junctions first (safe even if none exist)
            try:
                cur.execute(f"DELETE FROM project_documents WHERE doc_id = {placeholder}", (doc_id,))
            except Exception:
                # If the project_documents table doesn't exist or delete fails, continue
                pass

            # Delete the document record
            cur.execute(f"DELETE FROM documents WHERE doc_id = {placeholder}", (doc_id,))
            conn.commit()

            return cur.rowcount > 0
        except Exception as e:
            if conn:
                conn.rollback()
            raise Exception(f"Error deleting document: {e}")
        finally:
            if conn:
                conn.close()

    def update_project_details(self, project_id: str, name: str = None, description: str = None):
        """Update project details"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            if self.use_rds:
                # MySQL syntax for updating timestamp
                if name and description:
                    cur.execute(
                        f"UPDATE projects SET name = {placeholder}, description = {placeholder}, updated_at = CURRENT_TIMESTAMP WHERE project_id = {placeholder}",
                        (name, description, project_id)
                    )
                elif name:
                    cur.execute(
                        f"UPDATE projects SET name = {placeholder}, updated_at = CURRENT_TIMESTAMP WHERE project_id = {placeholder}",
                        (name, project_id)
                    )
                elif description:
                    cur.execute(
                        f"UPDATE projects SET description = {placeholder}, updated_at = CURRENT_TIMESTAMP WHERE project_id = {placeholder}",
                        (description, project_id)
                    )
            else:
                # SQLite syntax
                if name and description:
                    cur.execute(
                        "UPDATE projects SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ?",
                        (name, description, project_id)
                    )
                elif name:
                    cur.execute(
                        "UPDATE projects SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ?",
                        (name, project_id)
                    )
                elif description:
                    cur.execute(
                        "UPDATE projects SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE project_id = ?",
                        (description, project_id)
                    )
            conn.commit()
        finally:
            conn.close()

    def delete_project(self, project_id: str) -> bool:
        """Delete a project record (cascades to project_documents)"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"DELETE FROM projects WHERE project_id = {placeholder}", (project_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # Project-Document relationship methods
    def add_document_to_project(self, project_id: str, doc_id: str) -> bool:
        """Add a document to a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"INSERT INTO project_documents (project_id, doc_id) VALUES ({placeholder}, {placeholder})",
                (project_id, doc_id)
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            # Handle duplicate key error gracefully
            if "Duplicate" in str(e) or "UNIQUE constraint" in str(e):
                return False  # Already exists
            raise e
        finally:
            conn.close()

    def remove_document_from_project(self, project_id: str, doc_id: str) -> bool:
        """Remove a document from a project"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(
                f"DELETE FROM project_documents WHERE project_id = {placeholder} AND doc_id = {placeholder}",
                (project_id, doc_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def get_document_projects(self, doc_id: str) -> List[Project]:
        """Get all projects that contain a specific document"""
        conn = self.get_connection()
        cur = conn.cursor()
        placeholder = self._get_placeholder()
        
        try:
            cur.execute(f"""
                SELECT p.id, p.project_id, p.name, p.description, p.user_id, p.doc_ids, p.created_at, p.updated_at 
                FROM projects p
                INNER JOIN project_documents pd ON p.project_id = pd.project_id
                WHERE pd.doc_id = {placeholder}
                ORDER BY p.created_at DESC
            """, (doc_id,))
            rows = cur.fetchall()
            
            result = []
            for row in rows:
                # Parse doc_ids from JSON string if present
                doc_ids = None
                if row[5]:
                    try:
                        doc_ids = json.loads(row[5])
                    except json.JSONDecodeError:
                        doc_ids = [row[5]]  # Fallback to old single doc_id format
                
                result.append(Project(
                    id=row[0],
                    project_id=row[1],
                    name=row[2],
                    description=row[3],
                    user_id=row[4],
                    doc_ids=doc_ids,
                    created_at=row[6],
                    updated_at=row[7]
                ))
            
            return result
        finally:
            conn.close()

# Global database manager instance
db_manager = DatabaseManager()

# get_all_subscription_plans
