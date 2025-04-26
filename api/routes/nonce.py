from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from uuid import uuid4
import structlog

from api.middleware.error_handler import APIError
from api.schemas.validation import NonceRequest
from api.cache.redis_manager import get_redis

logger = structlog.get_logger()
nonce_bp = Blueprint('nonce', __name__, url_prefix='/api')


@nonce_bp.route("/nonce", methods=["POST"])
async def generate_nonce():
    try:
        request_data = NonceRequest(**request.get_json())
        nonce = str(uuid4())

        if request_data.prefix:
            nonce = f"{request_data.prefix}_{nonce}"

        if request_data.length:
            nonce = nonce[: request_data.length]

        # Store nonce in Redis
        redis = get_redis()
        redis.set(nonce, "1", ex=300)  # Store for 5 mins

        return jsonify({"nonce": nonce}), 200
    except ValidationError as e:
        raise APIError(str(e), status_code=422)
