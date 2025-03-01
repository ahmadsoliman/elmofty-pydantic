from flask import Flask, jsonify, request
from dotenv import load_dotenv
import os

if os.getenv("FLASK_ENV") != "testing":
    load_dotenv()

from api.pydantic_agent import run_agent
from api.telegram_bot import (
    send_reply,
    reply_start,
    reply_loading,
    delete_loading_message,
)

# import logfire

# Configure logfire to suppress warnings (optional)
# logfire.configure(send_to_logfire="never")

app = Flask(__name__)


# {
#     "message": "Why do we have to pray?",
#     "first_name": "Ahmad",
#     "last_name": "Soliman",
#     "user_id": "412",
#     "message_id": "124",
#     "chat_id": "123"
# }
# IslamQA AI Chatbot API Endpoint for mobile APP
@app.route("/api/chat", methods=["POST"])
async def chat():
    msg_request = request.get_json()
    if not msg_request or "message" not in msg_request:
        return jsonify({"error": "Missing message attribute"}), 422

    user_input = msg_request["message"]
    result = await run_agent(user_input)
    return jsonify(result), 200


# IslamQA AI Chatbot API Endpoint Webhook for telegram bot
@app.route("/api/telegram", methods=["POST"])
async def telegram():
    msg_request = request.get_json() or {}

    if "message" in msg_request:
        message = msg_request.get("message", {})
        user_input = message.get("text", "")
        chat_id = message.get("chat", {}).get("id", None)

        if not message or not chat_id:
            return "Invalid request format", 400

        is_bot = message.get("from", {}).get("is_bot", False)

        if is_bot:
            return "Bot message Ignored.", 200

        if user_input == "/start":
            reply_start(chat_id)
            return "Initiated Conversation", 200

        reply_loading_response = reply_loading(chat_id)
        loading_message = reply_loading_response.get("result", {})

        result = await run_agent(user_input)

        delete_loading_message(chat_id, loading_message.get("message_id", -1))

        send_reply(chat_id, result["telegram_mesasge"])

        return jsonify(result), 200

    return "Ignored. Please send message and chat id.", 200


# {
#     message: string;
#     issue: string;
#     reasons: string[];
# }
@app.route("/api/report", methods=["POST"])
def report():
    return "Reported", 200


if __name__ == "__main__":
    app.run(debug=True)
