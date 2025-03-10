import structlog
import cohere
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from api.db import match_documents, MatchResult, get_qas, QA

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
        responses: List[List[MatchResult]] = []
        with ThreadPoolExecutor() as executor:
            # Submit all RPC calls to be executed concurrently in threads
            futures = [
                executor.submit(match_documents, embedding) for embedding in embeddings
            ]
            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as rpc_error:
                    logger.error(f"RPC error: {rpc_error}")

        question_similarity = dict()
        for response in responses:
            for obj in response:
                question_similarity[obj.question_id] = max(
                    question_similarity.get(obj.question_id, float("-inf")),
                    obj.similarity,
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
