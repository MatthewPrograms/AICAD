"""Standalone GUI entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog

from autocad_mcp.gui.app import run_app


def _configure_logging() -> Path:
    level_name = os.environ.get("AUTOCAD_MCP_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    default_log_file = Path.cwd() / "logs" / "autocad_mcp_gui.log"
    log_file = Path(os.environ.get("AUTOCAD_MCP_LOG_FILE", str(default_log_file)))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    structlog.get_logger().info(
        "gui_logging_configured",
        log_file=str(log_file),
        log_level=level_name,
    )
    return log_file


def main() -> None:
    log_file = _configure_logging()
    os.environ["AUTOCAD_MCP_ACTIVE_LOG_FILE"] = str(log_file)
    run_app()


if __name__ == "__main__":
    main()
