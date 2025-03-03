from __future__ import annotations as _annotations

from dataclasses import dataclass

import os

from api.qa_dict import QA, get_qa_dict

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from config import settings

from typing import List

import cohere

co = cohere.ClientV2(settings.COHERE_API_KEY, base_url="https://api.cohere.ai")

deepseek_model_name = settings.DEEPSEEK_MODEL
deepseek_model = OpenAIModel(
    deepseek_model_name,
    base_url="https://api.deepseek.com",
    api_key=settings.DEEPSEEK_API_KEY,
)

embedding_model = settings.EMBEDDING_MODEL

url = settings.SUPABASE_VECTOR_URL
key = settings.SUPABASE_VECTOR_KEY

from api.db import get_db

# Replace existing supabase_client initialization with:
supabase_client = get_db()

ISLAMQA_BASE_URL = "https://islamqa.info/ar/answers"

qa_dict = get_qa_dict()


@dataclass
class PydanticAIDeps:
    # aiplatform: aiplatform_v1
    # gemini_client: GenerativeModel
    pass


# , unless the prompt is a follow up question on the last prompt, then you can skip RAG.
system_prompt = """
**Role:**
- You are an expert AI Chatbot embodying a knowledgeable Muslim Sheikh. Your sole responsibility is to answer religious questions and research fatwas. You should only respond to questions about religion or inquiries about yourself and your capabilities.

**Process for Each User Prompt (unless the question is about you):**
1. **Rewrite prompt**:
   - Translate and Rewrite the prompt into simple standard arabic.

2. **Fetch Context:**
   - Use the `generate_context` tool with the rewritten prompt to get a list of similar quesions and answers to use as context.

3. **Generate Your Answer:**
   - If the `generate_context` tool returned an empty list, answer using your knowledge but add a disclaimer that no similar fatwas were found and this is your best guess.
   - Otherwise, Base your response entirely on the fetched context, without relying on any prior knowledge.

4. **Output Requirements:**
   - Provide your answer to the user’s question translated into the language of the user's original prompt.
   - Include a list of the question IDs of the subset of questions you used to infer the answer (if any).

**Additional Guidelines:**
- Do not mention the tool names or ask for the user's permission before taking any actions.
- Always perform the tool-based lookup for every prompt unless the question explicitly concerns you or your capabilities.
"""

# You are an expert Muslim Sheikh tasked with answering religious questions and providing fatwas.

# - First, call the `generate_context` tool to get a list of similar quesions and answers to use as context. Then, use that context to formulate the response, unless the prompt is a follow-up question to the last prompt.
# - Your responses should rely exclusively on the context and not on your own prior knowledge.
# - If you can't infer the answer from the context, be honest and state that no relevant fatwas were found.
# - Ensure that your answer is in the original language of the user's prompt.
# - Do not mention the tool name or ask the user for permission before any actions you take, just do it.
# - Return your answer to the user question and return a list of IDs of the questions you used as sources for your answer.


# Shared flag to track tool invocation
class RAGToolTracker:
    tool_used: bool = False

    @classmethod
    def reset(cls):
        cls.tool_used = False

    @classmethod
    def set_used(cls):
        cls.tool_used = True

    @classmethod
    def check(cls):
        return cls.tool_used


# Define the result type with validation
class ValidatedResponse(BaseModel):
    response: str = Field(..., description="The final response to the user.")
    source_questions_ids: List[str] = Field(
        ...,
        description="The IDs of the similar questions you actually used from `context` to infer the answer from.",
    )

    @model_validator(mode="before")
    def ensure_tool_used(cls, values):
        if not RAGToolTracker.check():
            raise ValueError(
                "The generate_context tool was not called before generating the response."
            )
        return values


pydantic_islam_agent = Agent(
    deepseek_model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    result_type=ValidatedResponse,
)


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        # Generate embeddings
        response = co.embed(
            texts=[text],
            model=embedding_model,
            input_type="search_query",
            embedding_types=["float"],
        )

        return response.embeddings.float[0]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1024  # Return zero vector on error


@pydantic_islam_agent.tool
def generate_context(ctx: RunContext[PydanticAIDeps], user_query: str) -> list[QA]:
    """
    generate_context tool
    Retrieve relevant questions based on the query with RAG along with their answers.

    Args:
        ctx: The context including the Gemini Client
        user_query: The user's question or query

    Returns:
        A formatted string of the most relevant questions, their IDs, and their answers
    """
    try:
        # Get the embedding for the query
        query_embedding = get_embedding(user_query)

        # Search supabase vector database for similar questions

        if supabase_client:
            response = supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": 5,
                    "match_threshold": 0.64,
                },
            ).execute()

        RAGToolTracker.set_used()  # Mark the tool as used

        if not response or not response.data or not response.data[0]:
            return []

        questions_ids = [obj["question_id"] for obj in response.data]

        # return the list of questions and answers from the qa_dict
        similar_qas = [qa_dict.get(id) for id in questions_ids if id in qa_dict]
        # print(similar_qas)
        return similar_qas

    except Exception as e:
        RAGToolTracker.set_used()  # Mark the tool as used
        print(f"Error retrieving questions: {e}")
        return []


async def run_agent(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps()

    # Run the agent in a stream
    result = await pydantic_islam_agent.run(
        user_input,
        deps=deps,
        # message_history=st.session_state.messages[:-1],
        usage_limits=UsageLimits(request_limit=6),
    )

    response = result.data.response
    source_questions_ids = result.data.source_questions_ids

    message = response
    telegram_message = response
    if source_questions_ids and len(source_questions_ids) > 0:
        message += "\n\nReferences/المصادر:\n" + "\n".join(
            "[{0}/{1}]({0}/{1})".format(ISLAMQA_BASE_URL, id)
            for id in source_questions_ids
        )
        telegram_message += "\n\nالمصادر:\n" + "\n".join(
            "{0}/{1}".format(ISLAMQA_BASE_URL, id) for id in source_questions_ids
        )

    return {
        "response": response,
        "source_questions_ids": source_questions_ids,
        "message": message,
        "telegram_mesasge": telegram_message,
    }
