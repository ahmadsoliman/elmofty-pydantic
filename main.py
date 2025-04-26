from api import create_app
from dotenv import load_dotenv
from config import settings
import structlog

logger = structlog.get_logger()

# Load environment variables
if settings.FLASK_ENV != "testing":
    load_dotenv()

# Import create_app after environment setup

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(settings.PORT))
