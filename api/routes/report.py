from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import structlog

from api.middleware.error_handler import APIError
from api.middleware.play_integrity import verify_online
from api.schemas.validation import ReportRequest
from config import settings

logger = structlog.get_logger()
report_bp = Blueprint('report', __name__, url_prefix='/api')


@report_bp.route("/report", methods=["POST"])
def report():
    try:
        data = request.get_json()

        if settings.FLASK_ENV != "testing":
            if "integrity_token" in data:
                passedIntegrity = verify_online(data.get("integrity_token"))

            if not passedIntegrity:
                raise APIError(
                    "Integrity token isn't provided or is invalid", status_code=403
                )

        report_request = ReportRequest(**data)
        # save in redis and implement an endpoint to get all reports
    except ValidationError as e:
        raise APIError(str(e), status_code=422)

    # Process the report request
    return "Reported", 200
