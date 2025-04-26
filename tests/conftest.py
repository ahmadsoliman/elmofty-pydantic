import sys
import os

# Add the project directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dotenv import load_dotenv
import pytest
from unittest.mock import patch

# Load environment variables from .env.test file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env.test"))

# Import test settings
from tests.test_settings import test_settings

FLASK_ENV = "testing"


# Patch the settings module for all tests
@pytest.fixture(autouse=True, scope="session")
def mock_settings():
    with patch("config.settings", test_settings):
        with patch("api.agents.embedding.settings", test_settings):
            with patch("api.agents.response_agent.settings", test_settings):
                with patch("api.agents.translation_agent.settings", test_settings):
                    with patch("api.agents.orchesterator.settings", test_settings):
                        yield
