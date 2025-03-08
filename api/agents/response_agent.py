from __future__ import annotations as _annotations
from typing import List
import structlog

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import ModelMessage
from config import settings

from api.agents.embedding import NO_QUESTIONS_FOUND

logger = structlog.get_logger()

deepseek_model_name = settings.DEEPSEEK_MODEL
deepseek_model = OpenAIModel(
    deepseek_model_name,
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
)

# , unless the prompt is a follow up question on the last prompt, then you can skip RAG.
system_prompt = f"""
**Role:**
- You are an expert AI Chatbot embodying a knowledgeable Muslim Sheikh. Your sole responsibility is to answer religious questions. You should only respond to religious questions, and inquiries about yourself or your capabilities.

**Response (unless the question is about you):**
- Each prompt contains  "User question", "Resources", and "Target language"
- To understand the "User question", take into consideration the conversation history.
- Answer only in the "Target language"
- If the Resources says "{NO_QUESTIONS_FOUND}", answer using your knowledge but add a disclaimer that no similar questions were found and this is your best guess.
- Otherwise, Base your response entirely on the Resources, without relying on any prior knowledge.

**Output Requirements:**
- Provide your answer to the user question translated into the target language.
- Include a list of the question IDs of only the subset of questions you used to infer the answer (if any).
"""


# Define the result type with validation
class ValidatedResponse(BaseModel):
    response: str = Field(
        ...,
        description="The final response to the user translated into the target language.",
    )
    source_questions_ids: List[str] = Field(
        ...,
        description="The IDs of the similar questions you actually used from `resources` to infer the answer from.",
    )


pydantic_islam_agent = Agent(
    deepseek_model,
    system_prompt=system_prompt,
    retries=2,
    result_type=ValidatedResponse,
)


async def run_response_agent(
    user_input: str,
    context: str,
    target_language: str,
    conversation: list[ModelMessage] = [],
):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    try:
        prompt = f"User question: {user_input}\n\nResources:{context}\n\nTarget language: {target_language}"

        result = await pydantic_islam_agent.run(
            prompt,
            message_history=conversation,
            usage_limits=UsageLimits(request_limit=2),
        )

        return result.data
    except Exception as e:
        logger.error(f"Error inside pydantic agent: {e}")
        raise e
