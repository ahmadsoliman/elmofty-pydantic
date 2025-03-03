from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import ValidationError
from uuid import uuid4


from api.middleware.error_handler import register_error_handlers
from api.services.chat_service import ChatService
from api.services.telegram_service import TelegramService
from api.middleware.error_handler import APIError
from api.cache.redis_manager import get_redis
from api.services.play_integrity_service import verify_online
from api.logging_config import configure_logging
from config import settings

from api.schemas.validation import (
    ChatRequest,
    TelegramRequest,
    ReportRequest,
    NonceRequest,
)
from dotenv import load_dotenv


if settings.FLASK_ENV != "testing":
    load_dotenv()

app = Flask(__name__)
register_error_handlers(app)
CORS(app, resources={r"/*": {"origins": "*"}})
configure_logging()


@app.route("/api/chat", methods=["POST"])
async def chat():
    try:
        data = request.get_json()

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


# IslamQA AI Chatbot API Endpoint Webhook for telegram bot
@app.route("/api/telegram", methods=["POST"])
async def telegram():
    try:
        msg_request = TelegramRequest(**request.get_json())
        result = await TelegramService.process_telegram_request(msg_request)
        return jsonify(result), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)


@app.route("/api/report", methods=["POST"])
def report():
    try:
        report_request = ReportRequest(**request.get_json())
        # save in redis and implement an endpoint to get all reports
    except ValidationError as e:
        raise APIError(str(e), status_code=422)

    # Process the report request
    return "Reported", 200


@app.route("/api/nonce", methods=["POST"])
async def generate_nonce():
    try:
        request_data = NonceRequest(**request.get_json())
        nonce = str(uuid4())

        if request_data.prefix:
            nonce = f"{request_data.prefix}_{nonce}"

        if request_data.length:
            nonce = nonce[: request_data.length]

        # Store nonce in Redis
        redis = get_redis()
        redis.set(nonce, "1", ex=300)  # Store for 1 hour

        return jsonify({"nonce": nonce}), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)


if __name__ == "__main__":
    app.run(debug=True)
