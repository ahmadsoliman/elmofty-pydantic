import structlog
from time import time
from flask import request

logger = structlog.get_logger()

def log_requests(app):
    @app.before_request
    def before_request():
        request.start_time = time()

    @app.after_request
    def after_request(response):
        logger.info(
            "request",
            method=request.method,
            path=request.path,
            status=response.status_code,
            duration=(time() - request.start_time) * 1000,
            ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        return response
