"""Structured logging setup for ConsensusX."""

import logging
import sys


def get_logger(name: str = "consensus_x", level: str = "INFO") -> logging.Logger:
    """Returns a configured logger with clean formatting.
    
    Args:
        name: Name of the logger, typically __name__.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger

