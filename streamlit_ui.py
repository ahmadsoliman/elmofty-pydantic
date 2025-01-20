from __future__ import annotations
import asyncio
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
