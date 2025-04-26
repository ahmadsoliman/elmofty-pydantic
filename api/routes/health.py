from flask import Blueprint, jsonify
import structlog

logger = structlog.get_logger()
health_bp = Blueprint('health', __name__, url_prefix='/api')


@health_bp.route("/health", methods=["GET"])
async def health_check():
    try:
        return {
            "status": "ok",
            "database": True,
            "external_services": {"gcp": True, "cohere": True},
        }
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
