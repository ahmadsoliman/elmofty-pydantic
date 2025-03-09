from api.agents.orchesterator import process_user_input
from pydantic_ai.messages import ModelResponse, ModelRequest, TextPart, UserPromptPart
from api.schemas.validation import ChatRequest, ChatMessage

# from api.cache.cache_decorator import cache_response


class ChatService:
    @staticmethod
    # @cache_response(ttl=3600, key_of_arg=ChatRequest.hash)  # Cache for 1 hour
    async def process_chat_request(data: ChatRequest):
        user_input = data.message

        def init_msg(msg: ChatMessage):
            if msg.sender == "bot":
                return ModelResponse(parts=[TextPart(content=msg.text)])
            else:
                return ModelRequest(parts=[UserPromptPart(content=msg.text)])

        chat_history = [init_msg(msg) for msg in data.chat_history]
        return await process_user_input(user_input, chat_history)
