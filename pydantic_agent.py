from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

from qa_dict import QA, qa_dict

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, RunContext
from typing import List

from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    # aiplatform: aiplatform_v1
    # gemini_client: GenerativeModel
    pass


# , unless the prompt is a follow up question on the last prompt, then you can skip RAG.
system_prompt = """
You are an expert Muslim Sheikh tasked with answering religious questions and providing fatwas.

- First, call the `generate_context` tool to get a list of similar quesions and answers to use as context. Then, use that context to formulate the response, unless the prompt is a follow-up question to the last prompt.
- Your responses should rely exclusively on the context and not on your own prior knowledge. 
- If you can't infer the answer from the context, be honest and state that no relevant fatwas were found.
- Ensure that your answer is in the original language of the user's prompt.
- Do not mention the tool name or ask the user for permission before any actions you take, just do it.
- Return your answer to the user question and return a list of IDs of the questions you used as sources for your answer.
"""


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
    context: list[QA] = Field(
        ...,
        description="The similar questions and answers from the generate_context tool.",
    )
    source_questions_ids: List[str] = Field(
        ...,
        description="The IDs of the similar questions and answers you actually used from `context` to infer the answer from.",
    )

    @model_validator(mode="before")
    def ensure_tool_used(cls, values):
        if not RAGToolTracker.check():
            raise ValueError(
                "The generate_context tool was not called before generating the response."
            )
        return values


pydantic_islam_agent = Agent(
    model=llm_model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    result_type=ValidatedResponse,
)


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        # Generate embeddings
        response = client.embeddings.create(
            input=[text],
            model=embedding_model,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@pydantic_islam_agent.tool
def generate_context(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    generate_context tool
    Retrieve relevant questions based on the query with RAG along with their answers.

    Args:
        ctx: The context including the Gemini Client
        user_query: The user's question or query

    Returns:
        A formatted string of the top 5 most relevant questions, their IDs, and their answers
    """
    try:
        # Get the embedding for the query
        query_embedding = get_embedding(user_query)

        # Search supabase vector database for similar questions

        response = supabase.rpc(
            "match_documents", {"query_embedding": query_embedding, "match_count": 5}
        ).execute()

        # print the responses first row

        if not response.data or not response.data[0]:
            return "No relevant questions found."

        questions_ids = [obj["content"] for obj in response.data]

        RAGToolTracker.set_used()  # Mark the tool as used
        # return the list of questions and answers from the qa_dict
        similar_qas = [qa_dict.get(id) for id in questions_ids if id in qa_dict]
        # print(similar_qas)
        return similar_qas

        # formatted_questions = []
        # for id in questions_ids:
        #     qa = qa_dict.get(id)
        #     if qa:
        #         formatted_questions.append(
        #             f"({id})سؤال: {qa.question}\n  الإجابة: {qa.answer}"
        #         )

        #     # Join all chunks with a separator
        # RAGToolTracker.set_used()  # Mark the tool as used
        # return "\n\n---\n\n".join(formatted_questions)

    except Exception as e:
        print(f"Error retrieving questions: {e}")
        return []
