import pytest
from flask import Flask
from ipaddress import _BaseNetwork
from api.middleware.telegram_security import (
    validate_telegram_ip,
    load_telegram_ip_ranges,
)
from api.middleware.error_handler import APIError


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


def test_valid_telegram_ip(app):
    # Get actual Telegram IP ranges
    telegram_ranges = load_telegram_ip_ranges()

    # Test with a valid IP from the ranges
    valid_ip = str(next(iter(next(iter(telegram_ranges)).hosts())))
    with app.test_request_context(environ_base={"REMOTE_ADDR": valid_ip}):
        try:
            validate_telegram_ip()
        except APIError:
            pytest.fail("Valid Telegram IP was rejected")


def test_invalid_telegram_ip(app):
    # Test with an invalid IP
    with app.test_request_context(environ_base={"REMOTE_ADDR": "1.2.3.4"}):
        with pytest.raises(APIError) as exc_info:
            validate_telegram_ip()
        assert "Request not from Telegram servers" in str(exc_info.value)


def test_ipv6_telegram_ip(app):
    # Test with an IPv6 address if present in ranges
    telegram_ranges = load_telegram_ip_ranges()
    ipv6_ranges = [ip for ip in telegram_ranges if ip.version == 6]

    if ipv6_ranges:
        valid_ipv6 = str(next(iter(next(iter(ipv6_ranges)).hosts())))
        with app.test_request_context(environ_base={"REMOTE_ADDR": valid_ipv6}):
            try:
                validate_telegram_ip()
            except APIError:
                pytest.fail("Valid Telegram IPv6 address was rejected")


def test_loading_ip_ranges():
    ranges = load_telegram_ip_ranges()
    assert len(ranges) > 0, "No IP ranges were loaded"
    assert all(isinstance(ip, _BaseNetwork) for ip in ranges), "Invalid IP range format"
