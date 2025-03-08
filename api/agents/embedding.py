import structlog
import cohere
from typing import List
import asyncio

from config import settings
from api.db import get_db
from api.services.qa_dict import QA, get_qas

NO_QUESTIONS_FOUND = "No similar questions were found."

logger = structlog.get_logger()

embedding_model = settings.EMBEDDING_MODEL
co = cohere.ClientV2(settings.COHERE_API_KEY, base_url="https://api.cohere.ai")


def get_embedding(texts: List[str]) -> List[List[float]]:
    """Get embedding vector from OpenAI."""
    try:
        # Generate embeddings
        response = co.embed(
            texts=texts,
            model=embedding_model,
            input_type="search_query",
            embedding_types=["float"],
        )

        return response.embeddings.float
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return []  # Return zero vector on error [0] * 1024


async def get_similar_questions(queries: str) -> list[QA]:
    """
    Retrieve relevant questions based on the query with RAG along with their answers.

    Args:
        user_query: The user's question or query

    Returns:
        A formatted string of the most relevant questions, their IDs, and their answers
    """
    try:
        # Get the embedding for the query
        embeddings = get_embedding(queries)

        # Search supabase vector database for similar questions

        db = await get_db()

        async def execute_rpc(embedding):
            return await db.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_count": settings.VECTOR_MATCH_COUNT,
                    "match_threshold": settings.VECTOR_MATCH_THRESHOLD,
                },
            ).execute()

        responses = await asyncio.gather(
            *[execute_rpc(embedding) for embedding in embeddings]
        )

        question_similarity = dict()
        for response in responses:
            if response.data and isinstance(response.data, List):
                for obj in response.data:
                    id = obj["question_id"]
                    similarity = obj["similarity"]
                    question_similarity[id] = max(
                        question_similarity.get(id, float("-inf")), similarity
                    )

        # Convert dict to list of tuples sorted by similarity score in descending order
        sorted_questions = sorted(
            question_similarity.items(), key=lambda x: x[1], reverse=True
        )
        # Get just the question IDs from the most similar questions
        questions_ids = [id for id, _ in sorted_questions][
            : settings.VECTOR_MATCH_COUNT
        ]

        similar_qas = get_qas(list(questions_ids))
        # logger.info(similar_qas)
        return similar_qas
    except Exception as e:
        logger.error(f"Error retrieving questions: {e}")
        return []


async def generate_context(queries: str):
    qas = await get_similar_questions(queries)

    if len(qas) == 0:
        return NO_QUESTIONS_FOUND

    context = ""

    for qa in qas:
        context += f"**سؤال({qa.id}):** {qa.question}\n **الإجابة:** {qa.answer}\n\n"

    return context
