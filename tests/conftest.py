import sys
import os

# Add the project directory to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from dotenv import load_dotenv

# Load environment variables from .env.test file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env.test"))

FLASK_ENV = "testing"
