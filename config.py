from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FLASK_ENV: str = Field(default="development")
    SUPABASE_QA_URL: str = Field(...)
    SUPABASE_QA_KEY: str = Field(...)
    SUPABASE_VECTOR_URL: str = Field(...)
    SUPABASE_VECTOR_KEY: str = Field(...)
    COHERE_API_KEY: str = Field(...)
    EMBEDDING_MODEL: str = Field(...)
    TELEGRAM_BOT_TOKEN: str = Field(...)
    TELEGRAM_SECRET_TOKEN: str = Field(...)
    DEEPSEEK_API_KEY: str = Field(...)
    DEEPSEEK_MODEL: str = Field(...)
    REDIS_HOST: str = Field(...)
    REDIS_PORT: str = Field(...)
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(...)
    ANDROID_PACKAGE_NAME: str = Field(...)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
