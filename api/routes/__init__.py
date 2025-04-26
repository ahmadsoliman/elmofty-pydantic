from flask import Blueprint
from importlib import import_module


def register_blueprints(app):
    """Register all blueprint modules"""
    blueprints = [
        'api.routes.chat:chat_bp',
        'api.routes.telegram:telegram_bp',
        'api.routes.report:report_bp',
        'api.routes.nonce:nonce_bp',
        'api.routes.health:health_bp',
        'api.routes.static:static_bp'
    ]

    for blueprint_path in blueprints:
        module_name, blueprint_name = blueprint_path.split(':')
        module = import_module(module_name)
        app.register_blueprint(getattr(module, blueprint_name))
