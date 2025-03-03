import requests
from ipaddress import ip_network
from functools import lru_cache
from typing import List
from config import settings


@lru_cache(maxsize=1)
def load_telegram_ip_ranges() -> List[ip_network]:
    """Load Telegram IP ranges from official source"""
    try:
        response = requests.get(
            "https://core.telegram.org/resources/cidr.txt", timeout=10
        )
        response.raise_for_status()

        ip_ranges = [ip_network("127.0.0.1")]
        for line in response.text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                try:
                    ip_ranges.append(ip_network(line))
                except ValueError:
                    continue  # Skip invalid CIDR notations
        return ip_ranges
    except requests.RequestException as e:
        # Fallback to hardcoded ranges if network request fails
        return (
            ip_network(ip)
            for ip in [
                "91.108.56.0/22",
                "91.108.4.0/22",
                "91.108.8.0/22",
                "91.108.16.0/22",
                "91.108.12.0/22",
                "149.154.160.0/20",
                "91.105.192.0/23",
                "91.108.20.0/22",
                "185.76.151.0/24",
                "2001:b28:f23d::/48",
                "2001:b28:f23f::/48",
                "2001:67c:4e8::/48",
                "2001:b28:f23c::/48",
                "2a0a:f280::/32",
            ]
        )


from flask import request
from ipaddress import ip_address, ip_network
from api.middleware.error_handler import APIError

# Official Telegram IP ranges as of October 2024
TELEGRAM_IP_RANGES = load_telegram_ip_ranges()


def validate_telegram_ip():
    """Validate that request comes from Telegram's servers"""
    client_ip = ip_address(request.remote_addr)
    telegram_ranges = load_telegram_ip_ranges()

    if not any(client_ip in network for network in telegram_ranges):
        print("error", client_ip)
        raise APIError("Request not from Telegram servers", status_code=403)


def validate_telegram_secret():
    """Validate Telegram secret token header"""
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")

    if not secret_token or secret_token != settings.TELEGRAM_SECRET_TOKEN:
        raise APIError("Invalid Telegram secret token", status_code=403)
