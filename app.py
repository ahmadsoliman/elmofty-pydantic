from flask import Flask, request, jsonify
from pydantic import ValidationError

from api.middleware.error_handler import register_error_handlers
from api.services.chat_service import ChatService
from api.services.telegram_service import TelegramService
from api.middleware.error_handler import APIError
from config import settings

from api.schemas.validation import (
    ChatRequest,
    TelegramRequest,
    ReportRequest,
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


if __name__ == "__main__":
    app.run(debug=True)
