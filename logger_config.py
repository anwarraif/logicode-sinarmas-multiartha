import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Default log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/{name}_{timestamp}.log"
    else:
        log_file = f"{log_dir}/{log_file}"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_function_execution(logger):
    """Decorator to log function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Starting execution of {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Successfully completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator
