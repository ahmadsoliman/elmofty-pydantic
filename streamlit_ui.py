from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

# import nest_asyncio


import streamlit as st
import logfire
from arabic_support import support_arabic_text

# Support Arabic text alignment in all components
support_arabic_text(components=["input", "markdown", "textinput"])

from qa_dict import qa_dict

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)
from pydantic_agent import pydantic_islam_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from vertexai.preview.generative_models import GenerativeModel

llm = os.getenv("LLM_MODEL", "gemini-1.5-pro")
gemini_model = GenerativeModel("gemini-1.5-pro")

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps()

    # Run the agent in a stream
    result = await pydantic_islam_expert.run(
        user_input,
        deps=deps,
        # message_history=st.session_state.messages[:-1],
    )

    message_placeholder = st.empty()
    message_placeholder.markdown(result.data)

    # We'll gather partial text to show incrementally
    # Render partial text as it arrives
    # async for chunk in result.stream_text(delta=True):
    #     partial_text += chunk
    #     message_placeholder.markdown(partial_text)

    # print("pt: ", partial_text)

    # Now that the stream is finished, we have a final result.
    # Add new messages from this run, excluding user-prompt messages
    filtered_messages = [
        msg
        for msg in result.new_messages()
        if not (
            hasattr(msg, "parts")
            and any(part.part_kind == "user-prompt" for part in msg.parts)
        )
    ]
    st.session_state.messages.extend(filtered_messages)

    # Add the final response to the messages
    st.session_state.messages.append(
        ModelResponse(parts=[TextPart(content=result.data)])
    )


async def main():
    st.title("المفتي AI")
    st.write(
        "مرحبا بك في عالم الذكاء الاصطناعي. يمكنك أن تسألني عن رأي العلماء في أي سؤال يتعلق بالمعاملات المالية وما شابه وسأعمل أحسن ما بوسعي لأجد الإجابة في موسوعة الأسئلة والأجوبة لدي."
    )

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("ماذا على بالك اليوم؟")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent(user_input)


if __name__ == "__main__":
    # nest_asyncio.apply()
    asyncio.run(main())
