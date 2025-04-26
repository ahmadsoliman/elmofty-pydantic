from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import structlog

from api.middleware.error_handler import APIError
from api.middleware.telegram_security import validate_telegram_secret
from api.schemas.validation import TelegramRequest
from api.services.telegram_service import TelegramService
from config import settings

logger = structlog.get_logger()
telegram_bp = Blueprint('telegram', __name__, url_prefix='/api')


@telegram_bp.route("/telegram", methods=["POST"])
async def telegram():
    try:
        # Validate request security
        if settings.FLASK_ENV != "testing":
            # validate_telegram_ip()
            validate_telegram_secret()

        # Process request
        msg_request = TelegramRequest(**request.get_json())
        result = await TelegramService.process_telegram_request(msg_request)
        return jsonify(result), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)
