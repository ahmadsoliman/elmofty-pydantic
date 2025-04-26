from __future__ import annotations as _annotations
import os
from pydantic_ai.models.openai import OpenAIModel
from config import settings

# Get model configuration from environment variables with fallbacks
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", settings.OPENROUTER_MODEL)

# Create a reusable model instance
openrouter_model = OpenAIModel(
    OPENROUTER_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)
