import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict
# hey check 

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types gracefully."""
    
    def default(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to string representation."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            return str(obj)
        except Exception:
            return f"<unserializable object of type {type(obj).__name__}>"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with timestamp, level, component, message, and optional extra fields."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        
        # Add optional extra fields if present
        if hasattr(record, 'tokens'):
            log_entry['tokens'] = record.tokens
        if hasattr(record, 'latency'):
            log_entry['latency'] = record.latency
        if hasattr(record, 'cost'):
            log_entry['cost'] = record.cost
        
        # Add any other extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'message', 'exc_info', 
                          'exc_text', 'stack_info', 'tokens', 'latency', 'cost']:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, cls=SafeJSONEncoder)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with JSON formatting.
    
    Args:
        name: Typically __name__ from the calling module
        
    Returns:
        Configured logger instance that outputs structured JSON logs
        
    Usage:
        from src.observability.logging_config import get_logger
        log = get_logger(__name__)
        log.info("Processing request", extra={"tokens": 100, "latency": 0.5})
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger
