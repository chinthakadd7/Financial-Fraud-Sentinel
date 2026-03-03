"""
Structured Logging Utility for Fraud Detection System.

Provides JSON-formatted logging compatible with cloud logging systems
(CloudWatch, ELK, Splunk) for production monitoring and debugging.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.
    
    Compatible with log aggregation systems and provides structured data
    for easier parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from LoggerAdapter or extra={} in log calls
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "transaction_id"):
            log_data["transaction_id"] = record.transaction_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        # Add any other custom attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
                "exc_info", "exc_text", "request_id", "user_id", 
                "transaction_id", "duration_ms"
            ] and not key.startswith("_"):
                log_data[key] = value
        
        return json.dumps(log_data)



def setup_logger(
    name: str,
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure and return a logger with structured formatting.
    
    Args:
        name: Logger name (usually module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional path to log file for persistent logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    formatter = JSONFormatter()

    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        # Always use JSON for file logging
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with application defaults.
    
    Convenience function that uses settings from config module.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    try:
        from src.config import settings
        return setup_logger(
            name=name,
            level=settings.LOG_LEVEL,
            log_format=settings.LOG_FORMAT,
            log_file=settings.LOG_FILE
        )
    except ImportError:
        # Fallback if config not available
        return setup_logger(name=name)


class RequestLogger:
    """
    Context manager for logging request/response with timing.
    
    Usage:
        with RequestLogger(logger, request_id="123") as req_log:
            # Do work
            req_log.info("Processing transaction")
    """
    
    def __init__(self, logger: logging.Logger, request_id: str):
        self.logger = logger
        self.request_id = request_id
        self.start_time = None
        self.adapter = logging.LoggerAdapter(
            logger,
            {"request_id": request_id}
        )
    
    def __enter__(self):
        """Start timing and return logger adapter."""
        self.start_time = datetime.utcnow()
        return self.adapter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log completion time."""
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            self.adapter.info(
                f"Request completed",
                extra={"duration_ms": round(duration, 2)}
            )


# Create default loggers for different components
api_logger = get_logger("fraud_sentinel.api")
model_logger = get_logger("fraud_sentinel.model")
dashboard_logger = get_logger("fraud_sentinel.dashboard")
pipeline_logger = get_logger("fraud_sentinel.pipeline")


# Example usage
if __name__ == "__main__":
    """Test logging functionality."""
    
    # Test JSON logging
    logger_json = setup_logger("test_json", level="DEBUG", log_format="json")
    logger_json.debug("Debug message")
    logger_json.info("Info message", extra={"request_id": "test-123"})
    logger_json.warning("Warning message")
    logger_json.error("Error message")
    
    print("\n" + "="*50 + "\n")
    
    # Test text logging
    logger_text = setup_logger("test_text", level="DEBUG", log_format="text")
    logger_text.debug("Debug message")
    logger_text.info("Info message", extra={"request_id": "test-456"})
    logger_text.warning("Warning message")
    logger_text.error("Error message")
    
    print("\n" + "="*50 + "\n")
    
    # Test RequestLogger
    with RequestLogger(logger_json, "req-789") as log:
        log.info("Starting request processing")
        import time
        time.sleep(0.1)
        log.info("Request processed successfully")
