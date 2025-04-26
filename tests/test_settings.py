from pydantic import Field
from pydantic_settings import BaseSettings


class TestSettings(BaseSettings):
    FLASK_ENV: str = Field(default="testing")
    PORT: int = Field(default=8080)
    POSTGRES_INSTANCE_CONNECTION: str = Field(default="db")
    POSTGRES_DB_PASSWORD: str = Field(default="123")
    COHERE_API_KEY: str = Field(default="123")
    EMBEDDING_MODEL: str = Field(default="embed-multilingual-v3.0")
    TELEGRAM_BOT_TOKEN: str = Field(default="123")
    TELEGRAM_SECRET_TOKEN: str = Field(default="123")
    # OpenRouter configuration
    OPENROUTER_API_KEY: str = Field(default="123")
    OPENROUTER_MODEL: str = Field(default="anthropic/claude-3-opus:2024-05-07")
    VECTOR_MATCH_COUNT: int = Field(default=5)
    VECTOR_MATCH_THRESHOLD: float = Field(default=0.6)
    REDIS_HOST: str = Field(default="127.0.0.1")
    REDIS_PORT: str = Field(default="6379")
    REDIS_USER: str = Field(default="default")
    REDIS_PASSWORD: str = Field(default="123")
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(default="gcp-service-account.json")
    ANDROID_PACKAGE_NAME: str = Field(default="com.anonymous.IslamQA")

    model_config = {"env_file": ".env.test", "env_file_encoding": "utf-8"}


test_settings = TestSettings()
