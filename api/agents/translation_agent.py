from __future__ import annotations as _annotations
from typing import List
import structlog

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import ModelMessage
from config import settings


logger = structlog.get_logger()

deepseek_model_name = settings.DEEPSEEK_MODEL
deepseek_model = OpenAIModel(
    deepseek_model_name,
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
)

# , unless the prompt is a follow up question on the last prompt, then you can skip RAG.
system_prompt = """
**Role:**
- You are an expert translator and rewriter. Your job is to translate prompt to arabic and rewrite it in different terms.
   
**Response**
   - Determine the language of the user's prompt and if it is Arabic or not.
   - If the the prompt is not in Arabic, translate it to standard arabic.
   - Rewrite the prompt in three different ways taking into consideration message history and the intention behind the prompt.

**Output Requirements:**
   - A list of rewritten versions of the prompt.
   - A boolean that is true if and only if the prompt's language is Arabic.
   - The name of the language of the prompt.
"""


# Define the result type with validation
class TranslationValidatedResponse(BaseModel):
    rewritten: List[str] = Field(
        ...,
        description="Three different versions of the prompt rewritten in standard arabic.",
    )
    isArabic: bool = Field(
        ...,
        description="A boolean that is true if and only if the original prompt's language is Arabic.",
    )
    language: str = Field(..., description="The language name of the original prompt.")


translation_agent = Agent(
    deepseek_model,
    system_prompt=system_prompt,
    retries=2,
    result_type=TranslationValidatedResponse,
)


async def run_translate_agent(user_input: str, conversation: list[ModelMessage] = []):
    """
    Run the agent for the user_input prompt, while maintaining the conversation .
    """
    try:
        # Run the agent in a stream
        result = await translation_agent.run(
            user_input,
            message_history=conversation,
            usage_limits=UsageLimits(request_limit=1),
        )

        return result.data
    except Exception as e:
        logger.error(f"Error inside translation agent: {e}")
        raise e
