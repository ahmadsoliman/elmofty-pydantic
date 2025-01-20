from __future__ import annotations
import asyncio
import os
import nest_asyncio


import streamlit as st
import logfire
from arabic_support import support_arabic_text

# Support Arabic text alignment in all components
support_arabic_text(components=["input", "markdown", "textinput"])


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
from pydantic_agent import pydantic_islam_agent, PydanticAIDeps, RAGToolTracker

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from vertexai.preview.generative_models import GenerativeModel


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
    result = await pydantic_islam_agent.run(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    )
    response = result.data.response

    message_placeholder = st.empty()
    message_placeholder.markdown(response)

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
    st.session_state.messages.append(ModelResponse(parts=[TextPart(content=response)]))


async def main():
    st.title("ChatSheikh")
    st.write(
        "مرحبا بك في عالم الذكاء الاصطناعي. يمكنك أن تسألني عن رأي العلماء في أي سؤال يتعلق بالمعاملات المالية وما شابه وسأعمل أحسن ما بوسعي لأجد الإجابة في موسوعة الأسئلة والأجوبة لدي."
    )


nest_asyncio.apply()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"An error occurred: {e}")
