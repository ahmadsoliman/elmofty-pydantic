import pytest
from unittest.mock import patch, MagicMock
import json
from flask.testing import FlaskClient
import fakeredis
import redis

from main import app
from api.agents.response_agent import (
    pydantic_islam_agent,
    ValidatedResponse,
)

client = app.test_client()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch):
    # Create a FakeRedis instance
    fake_redis = fakeredis.FakeRedis()

    # Patch redis.Redis to return the FakeRedis instance
    monkeypatch.setattr(redis, "Redis", lambda *args, **kwargs: fake_redis)


@patch("api.services.chat_service.process_user_input")
def test_chat_endpoint_valid_request(mock_process):
    # Test valid request
    mock_process.return_value = {
        "response": "test response",
        "source_questions_ids": ["1", "2"],
        "message": "test response\n\nReferences:\n[islamqa.info/ar/answers/1](islamqa.info/ar/answers/1)\n[islamqa.info/ar/answers/2](islamqa.info/ar/answers/2)",
        "telegram_mesasge": "test response\n\nReferences:\nislamqa.info/ar/answers/1\nislamqa.info/ar/answers/2",
    }

    payload = {
        "message": "Why do we have to pray?",
        "first_name": "Ahmad",
        "last_name": "Soliman",
        "user_id": "412",
        "message_id": "124",
        "chat_id": "123",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    response_json = response.get_json()
    assert "message" in response_json
    assert "response" in response_json
    assert "source_questions_ids" in response_json


def test_chat_endpoint_missing_fields():
    # Test missing required fields
    payload = {
        "first_name": "Ahmad",
        "last_name": "Soliman",
        "user_id": "412",
        "message_id": "124",
        "chat_id": "123",
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422  # Expecting validation error


def test_telegram_endpoint_start_command():
    # Test /start command
    payload = {
        "message": {"text": "/start", "chat": {"id": "123"}, "from": {"is_bot": False}}
    }
    response = client.post("/api/telegram", json=payload)
    assert response.status_code == 200
    assert response.get_json()["response"] == "Initiated Conversation"


def test_telegram_endpoint_bot_message():
    # Test bot message should be ignored
    payload = {
        "message": {
            "text": "test message",
            "chat": {"id": "123"},
            "from": {"is_bot": True},
        }
    }
    response = client.post("/api/telegram", json=payload)
    assert response.status_code == 200
    assert response.get_json()["response"] == "Bot message Ignored."


def test_telegram_endpoint_invalid_structure():
    # Test invalid message structure
    payload = {"invalid": "structure"}
    response = client.post("/api/telegram", json=payload)
    assert response.status_code == 422


@patch("api.telegram_bot.requests.post")
@patch("api.services.telegram_service.process_user_input")
def test_telegram_endpoint_valid_request(mock_process, mock_requests):
    # Test valid request
    mock_process.return_value = {
        "response": "test response",
        "source_questions_ids": ["1", "2"],
        "message": "test response\n\nReferences:\n[islamqa.info/ar/answers/1](islamqa.info/ar/answers/1)\n[islamqa.info/ar/answers/2](islamqa.info/ar/answers/2)",
        "telegram_mesasge": "test response\n\nReferences:\nislamqa.info/ar/answers/1\nislamqa.info/ar/answers/2",
    }

    # This mock is only for reply_loading request
    mock_response = MagicMock()
    mock_response.json.return_value = MagicMock(result=MagicMock(message_id="123"))
    mock_requests.return_value = mock_response

    payload = {
        "message": {
            "text": "Why do we pray?",
            "chat": {"id": "123"},
            "from": {"is_bot": False},
        }
    }
    response = client.post("/api/telegram", json=payload)
    assert response.status_code == 200
    response_json = response.get_json()
    assert "telegram_mesasge" in response_json


def test_report_endpoint():
    # Test report endpoint
    payload = {
        "message": "test message",
        "issue": "test issue",
        "reasons": ["reason1", "reason2"],
    }
    response = client.post("/api/report", json=payload)
    assert response.status_code == 200
    assert response.text == "Reported"
