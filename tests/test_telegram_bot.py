import pytest
from api.telegram_bot import (
    send_reply,
    reply_start,
    reply_loading,
    delete_loading_message,
    TELEGRAM_BOT_TOKEN,
)
from unittest.mock import patch, MagicMock
import requests


@patch("api.telegram_bot.requests.post")
def test_send_reply_success(mock_requests):
    # Test successful message sending
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_requests.return_value = mock_response

    result = send_reply("123", "test message")
    assert result == {"ok": True}
    mock_requests.assert_called_once_with(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": "123", "text": "test message"},
    )


@patch("api.telegram_bot.requests.post")
def test_send_reply_failure(mock_requests):
    # Test message sending failure
    mock_requests.side_effect = requests.exceptions.RequestException("API error")

    result = send_reply("123", "test message")
    assert result == {"ok": False, "error": str("API error")}


@patch("api.telegram_bot.requests.post")
def test_reply_start(mock_requests):
    # Test start message reply
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_requests.return_value = mock_response

    result = reply_start("123")
    assert result == {"ok": True}
    mock_requests.assert_called_once_with(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={
            "chat_id": "123",
            "text": """مرحباً بك! أنا شيخ مسلم متخصص في الإجابة على الأسئلة الدينية وتقديم الفتاوى في مجال المعاملات المالية.

تفضل بطرح سؤالك، وسأبحث في الفتاوى المعتمدة لأقدم لك الإجابة المناسبة وفق الشريعة الإسلامية.

⚠️ تحذير: الردود مولّدة بواسطة الذكاء الاصطناعي وقد تحتوي على أخطاء.
""",
        },
    )


@patch("api.telegram_bot.requests.post")
def test_reply_loading(mock_requests):
    # Test loading message reply
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_requests.return_value = mock_response

    result = reply_loading("123")
    assert result == {"ok": True}
    mock_requests.assert_called_once_with(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": "123", "text": "جاري البحث عن إجابة..."},
    )


@patch("api.telegram_bot.requests.post")
def test_delete_loading_message_success(mock_requests):
    # Test successful message deletion
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True}
    mock_requests.return_value = mock_response

    result = delete_loading_message("123", "456")
    assert result == {"ok": True}
    mock_requests.assert_called_once_with(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage",
        json={"chat_id": "123", "message_id": "456"},
    )


@patch("api.telegram_bot.requests.post")
def test_delete_loading_message_failure(mock_requests):
    # Test message deletion failure
    mock_requests.side_effect = requests.exceptions.RequestException("API error")

    result = delete_loading_message("123", "456")
    assert result == {"ok": False, "error": str("API error")}
