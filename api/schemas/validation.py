from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
import bleach


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    message_id: str = Field(..., min_length=1, max_length=100)
    chat_id: str = Field(..., min_length=1, max_length=100)

    @field_validator("*")
    def sanitize_strings(cls, value):
        if isinstance(value, str):
            return bleach.clean(value, strip=True)
        return value


class TelegramRequest(BaseModel):
    message: dict = Field(...)

    @field_validator("message")
    def validate_message(cls, value):
        if not isinstance(value, dict):
            raise ValidationError("Message must be a dictionary")
        if "text" not in value or "chat" not in value:
            raise ValidationError("Message must contain text and chat")
        return value


class ReportRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    issue: str = Field(..., min_length=1, max_length=2000)
    reasons: list[str] = Field(..., min_length=1, max_length=10)

    @field_validator("*")
    def sanitize_strings(cls, value):
        if isinstance(value, str):
            return bleach.clean(value, strip=True)
        return value


class HealthCheckResponse(BaseModel):
    status: str
    database: bool
    external_services: dict


from uuid import UUID

class NonceRequest(BaseModel):
    length: Optional[int] = Field(None, ge=16, le=128)
    prefix: Optional[str] = Field(None, max_length=32)

    @field_validator("prefix")
    def sanitize_prefix(cls, value):
        if value:
            return bleach.clean(value, strip=True)
        return value

