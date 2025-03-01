from typing import Dict, Any
from api.pydantic_agent import run_agent
from api.schemas.validation import ChatRequest

from api.cache.cache_decorator import cache_response


class ChatService:
    @staticmethod
    @cache_response(ttl=3600)  # Cache for 1 hour
    async def process_chat_request(data: ChatRequest):
        user_input = data.message
        return await run_agent(user_input)
