from __future__ import annotations

import json
import logging
import sys
from typing import Optional

from app.infra.config import get_current_config


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in {"msg", "args", "exc_info", "exc_text", "stack_info", "message"}:
                continue
            if key not in log_record:
                log_record[key] = value

        return json.dumps(log_record)


class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs



def _get_log_level() -> int:
    level_str = str(get_current_config().system.log_level).upper()
    return getattr(logging, level_str, logging.INFO)



def setup_logging(
    name: Optional[str] = None,
    json_logs: Optional[bool] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_get_log_level())

    if logger.handlers:
        return logger

    use_json = json_logs if json_logs is not None else bool(get_current_config().system.json_logs)
    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger



def get_logger(name: str) -> logging.Logger:
    return setup_logging(name)



def get_context_logger(name: str, **context) -> ContextAdapter:
    logger = setup_logging(name)
    return ContextAdapter(logger, context)
