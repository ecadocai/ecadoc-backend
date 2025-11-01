"""
Configuration settings for the Floor Plan Agent API
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings"""
    
    # API Configuration
    APP_NAME = "Floorplan LangGraph Agent + RAG API"
    VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # OpenAI model to use for Chat/OpenAI clients (configurable via env var OPENAI_MODEL)
    OPENAI_MODEL = os.getenv('OPENAI_MODEL')
    
    # Google OAuth Configuration
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    
    # Email Configuration for verification
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    FROM_EMAIL = os.getenv('FROM_EMAIL', 'noreply@esticore.com')
    
    # Email verification settings
    VERIFICATION_TOKEN_EXPIRE_HOURS = int(os.getenv('VERIFICATION_TOKEN_EXPIRE_HOURS', 1))
    OTP_EXPIRE_MINUTES = int(os.getenv('OTP_EXPIRE_MINUTES', 5))
    FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')
    
    # LangSmith Configuration
    LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', 'false').lower() == 'true'
    LANGSMITH_ENDPOINT = os.getenv('LANGSMITH_ENDPOINT')
    LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
    LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT')
    
    # Roboflow Configuration
    ROBOFLOW_API_URL =  os.getenv('ROBOFLOW_API_URL')
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    ROBOFLOW_MODEL_ID =  os.getenv('ROBOFLOW_MODEL_ID')
    
    # Tavily Configuration
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    
    # Database Configuration
    DATABASE_NAME = "project.db"  # Legacy SQLite database
    
    # Database Connection Configuration
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = int(os.getenv('DB_PORT', 5432))  # Default to PostgreSQL port
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')

    # Stripe Configuration
    STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
    STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')
    STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')

    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    S3_PUBLIC_BASE_URL = os.getenv('S3_PUBLIC_BASE_URL')
    CLOUDFRONT_DISTRIBUTION_DOMAIN = os.getenv('CLOUDFRONT_DISTRIBUTION_DOMAIN')

    # JWT and Base URL
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    BASE_URL = os.getenv('BASE_URL')

    # Database type selection and validation
    USE_RDS = bool(DB_HOST and DB_NAME and DB_USER and DB_PASSWORD)
    IS_POSTGRES = DB_PORT == 5432
    IS_MYSQL = DB_PORT == 3306
    
    # PostgreSQL/pgvector Configuration
    PGVECTOR_ENABLED = os.getenv('PGVECTOR_ENABLED', 'true').lower() == 'true'
    VECTOR_DIMENSIONS = int(os.getenv('VECTOR_DIMENSIONS', 1536))  # OpenAI embedding dimensions
    VECTOR_INDEX_LISTS = int(os.getenv('VECTOR_INDEX_LISTS', 100))  # IVFFlat index parameter
    
    # Database Connection Pool Settings
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 10))
    DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', 20))
    DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', 30))
    DB_POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', 3600))  # 1 hour
    
    # Database Storage Configuration
    USE_DATABASE_STORAGE = USE_RDS and IS_POSTGRES and PGVECTOR_ENABLED
    USE_LOCAL_STORAGE = not USE_DATABASE_STORAGE
    
    # Directory Configuration
    DATA_DIR = os.getenv('DATA_DIR', os.path.abspath("data"))
    
    if USE_DATABASE_STORAGE:
        VECTORS_DIR = None
        OUTPUT_DIR = None
        DOCS_DIR = None
        IMAGES_DIR = os.path.join(DATA_DIR, "images")
        print("INFO: Using PostgreSQL database storage - local file directories disabled")
    else:
        VECTORS_DIR = os.path.join(DATA_DIR, "vectors")
        OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
        DOCS_DIR = os.path.join(DATA_DIR, "docs")
        IMAGES_DIR = os.path.join(DATA_DIR, "images")
        print("INFO: Using local file storage - all directories enabled")
    
    # Security Configuration
    PASSWORD_MIN_LENGTH = 8
    
    # Agent Configuration
    RECURSION_LIMIT = 25
    CHAT_RECURSION_LIMIT = 20
    CHAT_HISTORY_LIMIT = 20
    SESSION_CLEANUP_HOURS = 24
    
    # Enhanced Session Management Configuration
    SESSION_CACHE_MAX_SIZE = int(os.getenv('SESSION_CACHE_MAX_SIZE', 1000))
    SESSION_MAINTENANCE_INTERVAL = int(os.getenv('SESSION_MAINTENANCE_INTERVAL', 3600))
    SESSION_ACTIVITY_UPDATE_PROBABILITY = float(os.getenv('SESSION_ACTIVITY_UPDATE_PROBABILITY', 0.01))
    SESSION_CLEANUP_PROBABILITY = float(os.getenv('SESSION_CLEANUP_PROBABILITY', 0.01))
    
    # Context Validation Settings
    ENABLE_STRICT_CONTEXT_VALIDATION = os.getenv('ENABLE_STRICT_CONTEXT_VALIDATION', 'true').lower() == 'true'
    ALLOW_CONTEXT_SWITCHING = os.getenv('ALLOW_CONTEXT_SWITCHING', 'true').lower() == 'true'
    
    # Session Security Settings
    SESSION_ACCESS_VALIDATION_ENABLED = os.getenv('SESSION_ACCESS_VALIDATION_ENABLED', 'true').lower() == 'true'
    SESSION_EXPIRY_GRACE_PERIOD_HOURS = int(os.getenv('SESSION_EXPIRY_GRACE_PERIOD_HOURS', 1))
    
    # File Configuration
    FILE_DELETE_DELAY = 3600
    CHAT_FILE_DELETE_DELAY = 7200
    
    # Database Migration Configuration
    MIGRATION_BATCH_SIZE = int(os.getenv('MIGRATION_BATCH_SIZE', 100))
    MIGRATION_PROGRESS_INTERVAL = int(os.getenv('MIGRATION_PROGRESS_INTERVAL', 10))
    ENABLE_MIGRATION_ROLLBACK = os.getenv('ENABLE_MIGRATION_ROLLBACK', 'true').lower() == 'true'
    
    # Database Maintenance Configuration
    AUTO_VACUUM_ENABLED = os.getenv('AUTO_VACUUM_ENABLED', 'true').lower() == 'true'
    VACUUM_SCHEDULE_HOURS = int(os.getenv('VACUUM_SCHEDULE_HOURS', 24))
    CLEANUP_ORPHANED_RECORDS = os.getenv('CLEANUP_ORPHANED_RECORDS', 'true').lower() == 'true'
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 100))
    
    # Performance Configuration
    ENABLE_QUERY_LOGGING = os.getenv('ENABLE_QUERY_LOGGING', 'false').lower() == 'true'
    SLOW_QUERY_THRESHOLD_MS = int(os.getenv('SLOW_QUERY_THRESHOLD_MS', 1000))
    ENABLE_CONNECTION_POOLING = os.getenv('ENABLE_CONNECTION_POOLING', 'true').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate required settings"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        if cls.USE_RDS:
            if not all([cls.DB_HOST, cls.DB_NAME, cls.DB_USER, cls.DB_PASSWORD]):
                raise ValueError("Database configuration incomplete. Please set DB_HOST, DB_NAME, DB_USER, and DB_PASSWORD")
            
            if cls.IS_POSTGRES:
                print(f"✓ Using PostgreSQL database: {cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
                if cls.PGVECTOR_ENABLED:
                    print(f"✓ pgvector enabled with {cls.VECTOR_DIMENSIONS} dimensions")
                    print("✓ Database storage enabled - files and vectors stored in PostgreSQL")
                else:
                    print("⚠ pgvector disabled - using local file storage")
            elif cls.IS_MYSQL:
                print(f"✓ Using MySQL database: {cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
                print("ℹ MySQL detected - using local file storage (pgvector not available)")
            else:
                print(f"⚠ Unknown database type on port {cls.DB_PORT} - assuming local file storage")
            
            if cls.DB_POOL_SIZE < 1:
                raise ValueError("DB_POOL_SIZE must be at least 1")
            if cls.DB_POOL_TIMEOUT < 1:
                raise ValueError("DB_POOL_TIMEOUT must be at least 1 second")
        else:
            print("ℹ Using SQLite database with local file storage")
        
        if cls.USE_DATABASE_STORAGE:
            if cls.VECTOR_DIMENSIONS not in [1536, 1024, 768, 512]:
                print(f"⚠ Unusual vector dimensions: {cls.VECTOR_DIMENSIONS}")
            if cls.VECTOR_INDEX_LISTS < 1:
                raise ValueError("VECTOR_INDEX_LISTS must be at least 1")
        
        print("DEBUG: Environment information:")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  DATA_DIR environment variable: {os.getenv('DATA_DIR', 'Not set')}")
        print(f"  Script location: {os.path.abspath(__file__)}")
        
        directories = {'DATA_DIR': cls.DATA_DIR}
        if cls.USE_DATABASE_STORAGE:
            directories['IMAGES_DIR'] = cls.IMAGES_DIR
        else:
            directories.update({
                'VECTORS_DIR': cls.VECTORS_DIR,
                'OUTPUT_DIR': cls.OUTPUT_DIR,
                'DOCS_DIR': cls.DOCS_DIR,
                'IMAGES_DIR': cls.IMAGES_DIR
            })
        
        print("DEBUG: Directory configuration:")
        for name, directory in directories.items():
            if directory:
                os.makedirs(directory, exist_ok=True)
                print(f"  ✓ {name}: {directory}")
                print(f"    exists: {os.path.exists(directory)}")
                print(f"    writable: {os.access(directory, os.W_OK)}")
            else:
                print(f"  - {name}: Disabled (using database storage)")
        
        print("\nStorage Configuration Summary:")
        print(f"  Database Storage: {'✓ Enabled' if cls.USE_DATABASE_STORAGE else '✗ Disabled'}")
        print(f"  Local File Storage: {'✓ Enabled' if cls.USE_LOCAL_STORAGE else '✗ Disabled'}")
        print(f"  Vector Storage: {'PostgreSQL/pgvector' if cls.USE_DATABASE_STORAGE else 'Local FAISS files'}")
        print(f"  File Storage: {'PostgreSQL binary' if cls.USE_DATABASE_STORAGE else 'Local filesystem'}")

# Global settings instance
settings = Settings()