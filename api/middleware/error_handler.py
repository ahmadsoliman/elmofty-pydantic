from flask import jsonify
import structlog
from typing import Dict, Any
from werkzeug.exceptions import HTTPException

logger = structlog.get_logger()


class APIError(Exception):
    def __init__(
        self, message: str, status_code: int = 500, details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def handle_error(error: Exception):
    if isinstance(error, APIError):
        response = jsonify(
            {
                "error": error.message,
                "details": error.details,
                "status": error.status_code,
            }
        )
        response.status_code = error.status_code
    elif isinstance(error, HTTPException):
        response = jsonify({"error": error.description, "status": error.code})
        response.status_code = error.code
    else:
        logger.exception("Unhandled exception occurred")
        response = jsonify({"error": "An unexpected error occurred", "status": 500})
        response.status_code = 500

    return response


def register_error_handlers(app):
    app.register_error_handler(APIError, handle_error)
    app.register_error_handler(HTTPException, handle_error)
    app.register_error_handler(Exception, handle_error)
