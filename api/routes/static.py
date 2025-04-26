from flask import Blueprint, jsonify, send_from_directory
import structlog

logger = structlog.get_logger()
static_bp = Blueprint('static', __name__)


@static_bp.route("/app-ads.txt")
def serve_static_file():
    try:
        return send_from_directory("static", "app-ads.txt")
    except FileNotFoundError:
        logger.error("app-ads.txt file not found in static directory")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error serving app-ads.txt: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
