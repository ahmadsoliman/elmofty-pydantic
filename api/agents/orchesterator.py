from __future__ import annotations as _annotations
import structlog

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage
from config import settings

from api.agents.translation_agent import run_translate_agent
from api.agents.response_agent import run_response_agent
from api.agents.embedding import generate_context
import time

logger = structlog.get_logger()

deepseek_model_name = settings.DEEPSEEK_MODEL
deepseek_model = OpenAIModel(
    deepseek_model_name,
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
)

ISLAMQA_BASE_URL = "https://islamqa.info/ar/answers"


async def process_user_input(user_input: str, conversation: list[ModelMessage] = []):
    """
    Run the agent the user_input prompt, while maintaining the entire conversation.
    """
    try:
        start_total = time.time()

        # translate to arabic and rewrite user question into different variants
        start = time.time()
        translation_result = await run_translate_agent(user_input, conversation)
        translation_time = time.time() - start

        # generate context by getting the most similar questions to the variants
        start = time.time()
        context = await generate_context(translation_result.rewritten)
        context_time = time.time() - start

        # answer the user question using context in the original language of the user question
        start = time.time()
        response_result = await run_response_agent(
            user_input, context, translation_result.language, conversation
        )
        response_time = time.time() - start

        total_time = time.time() - start_total
        logger.info(
            "Operation times",
            translation_time_ms=round(translation_time * 1000),
            translation_pct=round(translation_time / total_time * 100),
            context_time_ms=round(context_time * 1000),
            context_pct=round(context_time / total_time * 100),
            response_time_ms=round(response_time * 1000),
            response_pct=round(response_time / total_time * 100),
            total_time_ms=round(total_time * 1000),
        )

        # format responses
        response = response_result.response
        source_questions_ids = response_result.source_questions_ids

        message = response
        telegram_message = response
        if source_questions_ids and len(source_questions_ids) > 0:
            message += (
                f"\n\n{("References", "المصادر")[translation_result.isArabic]}:\n"
                + "\n".join(
                    "[{0}/{1}]({0}/{1})".format(ISLAMQA_BASE_URL, id)
                    for id in source_questions_ids
                )
            )
            telegram_message += (
                f"\n\n{("References", "المصادر")[translation_result.isArabic]}:\n"
                + "\n".join(
                    "{0}/{1}".format(ISLAMQA_BASE_URL, id)
                    for id in source_questions_ids
                )
            )

        return {
            "response": response,
            "source_questions_ids": source_questions_ids,
            "message": message,
            "telegram_mesasge": telegram_message,
        }
    except Exception as e:
        logger.error(f"Error processing user input in orchesterator: {e}")
        raise e
