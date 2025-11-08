"""
Celery application configured to use AWS SQS as the broker.
"""
import os
from celery import Celery
from modules.config.settings import settings
from modules.config.logger import logger


def _broker_url() -> str:
    # Celery uses the special transport 'sqs://' and boto3 AWS credentials
    return os.getenv("CELERY_BROKER_URL", "sqs://")


celery_app = Celery("ecadoc")

# Basic config for SQS
celery_app.conf.update(
    broker_url=_broker_url(),
    broker_transport_options={
        "region": settings.AWS_REGION or os.getenv("AWS_REGION", "us-east-1"),
        "visibility_timeout": int(os.getenv("SQS_VISIBILITY_TIMEOUT", "7200")),  # 2h for large PDFs
        "queue_name_prefix": os.getenv("SQS_QUEUE_PREFIX", "ecadoc-"),
        "polling_interval": float(os.getenv("SQS_POLLING_INTERVAL", "1")),
        "wait_time_seconds": int(os.getenv("SQS_WAIT_TIME_SECONDS", "10")),
    },
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "7200")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "6900")),
)

logger.info("celery_configured", extra={"broker": celery_app.conf.broker_url})

