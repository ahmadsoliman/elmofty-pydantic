from typing import Dict, Any
from api.pydantic_agent import run_agent
from api.schemas.validation import TelegramRequest

from api.cache.cache_decorator import cache_response
from api.telegram_bot import (
    reply_loading,
    reply_start,
    delete_loading_message,
    send_reply,
    reply_error,
)


class TelegramService:
    @staticmethod
    @cache_response(ttl=3600)  # Cache for 1 hour
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

        result = None
        try:
            result = await run_agent(user_input)
        except Exception as e:
            pass

        await delete_loading_message(chat_id, loading_message.get("message_id", -1))

        if result:
            send_reply(chat_id, result["telegram_mesasge"])
        else:
            reply_error(chat_id)

        return result
