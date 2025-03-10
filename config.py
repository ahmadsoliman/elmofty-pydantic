from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FLASK_ENV: str = Field(default="development")
    PORT: int = Field(default=8080)
    SUPABASE_QA_URL: str = Field(default="https://suujktweayfbbuwmcxai.supabase.co")
    SUPABASE_QA_KEY: str = Field(...)
    POSTGRES_INSTANCE_CONNECTION: str = Field(...)
    POSTGRES_DB_PASSWORD: str = Field(...)
    COHERE_API_KEY: str = Field(...)
    EMBEDDING_MODEL: str = Field(default="embed-multilingual-v3.0")
    TELEGRAM_BOT_TOKEN: str = Field(...)
    TELEGRAM_SECRET_TOKEN: str = Field(...)
    DEEPSEEK_API_KEY: str = Field(...)
    DEEPSEEK_MODEL: str = Field(default="deepseek-chat")
    VECTOR_MATCH_COUNT: int = Field(default=5)
    VECTOR_MATCH_THRESHOLD: float = Field(default=0.6)
    REDIS_HOST: str = Field(default="127.0.0.1")
    REDIS_PORT: str = Field(default="6379")
    REDIS_USER: str = Field(default="default")
    REDIS_PASSWORD: str = Field(...)
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(default="gcp-service-account.json")
    ANDROID_PACKAGE_NAME: str = Field(default="com.anonymous.IslamQA")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
