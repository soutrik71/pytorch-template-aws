import sys
import os
from pathlib import Path
from functools import wraps

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn


def setup_logger(log_file):
    """Set up the logger with a file and console handler."""
    os.makedirs(Path(log_file).parent, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(log_file, rotation="1MB")


def task_wrapper(func):
    """Wrapper to log the start and end of a task."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func_name}: {str(e)}")
            raise

    return wrapper


def get_rich_progress():
    """Get a Rich Progress object."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )
