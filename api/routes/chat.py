from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import structlog

from api.middleware.error_handler import APIError
from api.middleware.play_integrity import verify_online
from api.schemas.validation import ChatRequest
from api.services.chat_service import ChatService
from config import settings

logger = structlog.get_logger()
chat_bp = Blueprint('chat', __name__, url_prefix='/api')


@chat_bp.route("/chat", methods=["POST"])
async def chat():
    try:
        data = request.get_json()

        if settings.FLASK_ENV != "testing":
            passedIntegrity = False
            if "integrity_token" in data:
                passedIntegrity = verify_online(data.get("integrity_token"))

            if not passedIntegrity:
                raise APIError(
                    "Integrity token isn't provided or is invalid", status_code=403
                )

        msg_request = ChatRequest(**data)
        result = await ChatService.process_chat_request(msg_request)
        return jsonify(result), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)
