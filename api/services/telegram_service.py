from api.agents.orchesterator import process_user_input
from api.schemas.validation import TelegramRequest

from api.telegram_bot import (
    reply_loading,
    reply_start,
    delete_loading_message,
    send_reply,
    reply_error,
)


class TelegramService:
    async def process_telegram_request(data: TelegramRequest):
        message = data.message
        user_input = message.get("text", "")
        chat_id = message.get("chat", {}).get("id", None)

        is_bot = message.get("from", {}).get("is_bot", False)

        if is_bot:
            return {"response": "Bot message Ignored."}

        if user_input == "/start":
            reply_start(chat_id)
            return {"response": "Initiated Conversation"}

        reply_loading_response = reply_loading(chat_id)
        loading_message = reply_loading_response.get("result", {})

        try:
            result = await process_user_input(user_input)
            delete_loading_message(chat_id, loading_message.get("message_id", -1))
            send_reply(chat_id, result["telegram_mesasge"])

            return result
        except Exception as e:
            delete_loading_message(chat_id, loading_message.get("message_id", -1))
            reply_error(chat_id)
            raise e
