from flask import Flask, request, jsonify
from pydantic import ValidationError
from uuid import uuid4

from api.middleware.error_handler import register_error_handlers
from api.services.chat_service import ChatService
from api.services.telegram_service import TelegramService
from api.middleware.error_handler import APIError
from api.cache.redis_manager import get_redis
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


@app.route("/api/chat", methods=["POST"])
async def chat():
    try:
        msg_request = ChatRequest(**request.get_json())
        print(msg_request)
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
        redis.set(nonce, "1", ex=3600)  # Store for 1 hour

        return jsonify({"nonce": nonce}), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)


if __name__ == "__main__":
    app.run(debug=True)
