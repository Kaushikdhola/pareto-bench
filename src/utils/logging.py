"""Rich-based structured logger for pareto-bench."""

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger that writes formatted output via Rich.

    Calling this function multiple times with the same *name* returns the
    cached logger — handlers are not duplicated.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level; defaults to ``logging.INFO``.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)

    logger.propagate = False
    _loggers[name] = logger
    return logger
