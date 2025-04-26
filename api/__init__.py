from flask import Flask
from flask_cors import CORS
import structlog

from api.middleware.error_handler import register_error_handlers
from api.logging_config import configure_logging
from api.middleware.request_logger import log_requests
from api.routes import register_blueprints

logger = structlog.get_logger()


def create_app():
    """Flask application factory"""
    app = Flask(__name__)

    # Configure app
    register_error_handlers(app)
    CORS(app, resources={r"/*": {"origins": "*"}})
    configure_logging()
    log_requests(app)

    # Register blueprints
    register_blueprints(app)

    return app
