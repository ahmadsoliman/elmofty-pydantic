from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

from qa_dict import qa_dict

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from typing import List

from google.cloud import aiplatform_v1
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel

load_dotenv()

llm = os.getenv("LLM_MODEL", "gemini-1.5-pro")
model = GeminiModel(llm)

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    # aiplatform: aiplatform_v1
    # gemini_client: GenerativeModel
    pass


# , unless the prompt is a follow up question on the last prompt, then you can skip RAG.
system_prompt = """
You are an expert Muslim Sheikh tasked with answering religious questions and providing fatwas.

- First, call the `generate_context` tool to create a context string. Then, use that context to formulate the response, unless the prompt is a follow-up question to the last prompt.
- Your responses should rely exclusively on the context and not on your own prior knowledge. 
- If you can't infer the answer from the context, be honest and state that no relevant fatwas were found.
- Ensure that your answer is in the original language of the user's prompt.
- Do not mention the tool name or ask the user for permission before any actions you take, just do it.
- Include the sources of your answers at the end of each response by citing the full Q&As you referenced as is.
"""

# You ALWAYS use the tool retrieve_relevant_questions_with_answers for each prompt to find relevant questions and answers that you can infer the answer from.
# Don't mention the tool name in your answer.

# أنت شيخ مسلم خبير في الرد على الإستفسارات وإعطاء الفتاوى من خلال أداة البحث عن أسئلة مماثلة.
# لا تجيب على أسئلة أخرى خارج نطاق الدين الإسلامي وأحكامه، بخلاف وصف ما يمكنك القيام به.
# لا تسأل المستخدم قبل اتخاذ إجراء، قم به مباشرة.
# استنتج الإجابة فقط من إجابات الأسئلة المشابهة وليس من معرفتك السابقة.
# وإذا لم تجد إجابة في الأسئلة المشابهة، قل للمستخدم أن ليس لديك إجابة قاطعة - كن صادقًا.
# تأكد من تضمين السؤال والجواب الأصلي كما هو الذي وجدت منه الإجابة في نهاية ردك.

# YOU MUST USE THE TOOL retrieve_relevant_questions_with_answers EVERY TIME TO FIND THE ANSWER.
# ------------------------------------------------------------------------------------------
# You ALWAYS use the tool retrieve_relevant_questions_with_answers for each prompt to find relevant questions and answers that you can infer the answer from.
# Don't mention the tool name in your answer.
# You are an expert Muslim Sheikh who answers questions and gives fatwas using the similar questions and answers as context.

# Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

# Don't ask the user before taking an action, just do it.
# Only infer the answer from the similar questions' answers and not from your previous knowledge.
# Make sure to include the Q&A you found the answer from at the end of your response.

# When you first look for an answer, always start with RAG; unless you think the user's prompt is a follow up question to the previous one, then you can skip RAG.

# Always let the user know if you didn't find the answer in the RAG Q&As - be honest.
# Always answer using the user's prompt's original language.
# Always make sure you look for similar questions to the user's prompt using the provided tools before answering the user's question.
# YOU MUST USE THE TOOL retrieve_relevant_questions_with_answers EVERY TIME TO FIND THE ANSWER.

# """


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
    context_used: str = Field(
        ..., description="The generated context string from the generate_context tool."
    )

    @model_validator(mode="before")
    def ensure_tool_used(cls, values):
        if not RAGToolTracker.check():
            raise ValueError(
                "The generate_context tool was not called before generating the response."
            )
        return values


pydantic_islam_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2,
    result_type=ValidatedResponse,
)


# # Define the agent's logic
# @pydantic_islam_agent
# async def respond_with_context(
#     ctx: RunContext[PydanticAIDeps], user_input: str
# ) -> ValidatedResponse:
#     context = await ctx.call_tool("generate_context", user_input=user_input)
#     response = f"Using the context: {context}, here's your answer."
#     return ValidatedResponse(response=response, context_used=context)


def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")
        response = model.get_embeddings([text])
        return response[0].values
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 786  # Return zero vector on error


@pydantic_islam_agent.tool
def generate_context(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    generate_context tool
    Retrieve relevant questions based on the query with RAG along with their answers.

    Args:
        ctx: The context including the Gemini Client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 5 most relevant questions and their answers
    """
    try:
        # Get the embedding for the query
        query_embedding = get_embedding(user_query)

        # Query VertexAI for relevant questions
        client_options = {"api_endpoint": os.getenv("API_ENDPOINT")}
        vector_search_client = aiplatform_v1.MatchServiceClient(
            client_options=client_options
        )

        datapoint = aiplatform_v1.IndexDatapoint(feature_vector=query_embedding)
        query = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=5,
        )
        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=os.getenv("INDEX_ENDPOINT"),
            deployed_index_id=os.getenv("DEPLOYED_INDEX_ID"),
            queries=[query],
            return_full_datapoint=False,
        )
        response = vector_search_client.find_neighbors(request)
        print(response.nearest_neighbors[0].neighbors)

        if (
            not response.nearest_neighbors[0]
            or not response.nearest_neighbors[0].neighbors
        ):
            return "No relevant questions found."

        questions_ids = [
            obj.datapoint.datapoint_id
            for obj in response.nearest_neighbors[0].neighbors
        ]

        formatted_questions = []
        for id in questions_ids:
            qa = qa_dict.get(id)
            if qa:
                formatted_questions.append(
                    f"({id})سؤال: {qa['question']}\n  الإجابة: {qa['answer']}"
                )

            # Join all chunks with a separator
        RAGToolTracker.set_used()  # Mark the tool as used
        return "\n\n---\n\n".join(formatted_questions)

    except Exception as e:
        print(f"Error retrieving questions: {e}")
        return f"Error retrieving questions: {str(e)}"


# @pydantic_islam_expert.tool
# async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
#     """
#     Retrieve the full content of a specific documentation page by combining all its chunks.

#     Args:
#         ctx: The context including the Supabase client
#         url: The URL of the page to retrieve

#     Returns:
#         str: The complete page content with all chunks combined in order
#     """
#     try:
#         # Query Supabase for all chunks of this URL, ordered by chunk_number
#         result = (
#             ctx.deps.supabase.from_("site_pages")
#             .select("title, content, chunk_number")
#             .eq("url", url)
#             .eq("metadata->>source", "pydantic_ai_docs")
#             .order("chunk_number")
#             .execute()
#         )

#         if not result.data:
#             return f"No content found for URL: {url}"

#         # Format the page with its title and all chunks
#         page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
#         formatted_content = [f"# {page_title}\n"]

#         # Add each chunk's content
#         for chunk in result.data:
#             formatted_content.append(chunk["content"])

#         # Join everything together
#         return "\n\n".join(formatted_content)

#     except Exception as e:
#         print(f"Error retrieving page content: {e}")
#         return f"Error retrieving page content: {str(e)}"
