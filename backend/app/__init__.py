"""Flask application factory."""
from __future__ import annotations

import logging

from flask import Flask
from flask_cors import CORS

from app.config import Config, get_config


def create_app(config: type[Config] | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config or get_config())

    _configure_logging(app)
    _configure_extensions(app)
    _ensure_runtime_dirs(app)
    _register_services(app)
    _register_blueprints(app)
    _register_error_handlers(app)

    return app


def _register_services(app: Flask) -> None:
    if app.config["TESTING"]:
        return
    from app.services.chat_service import build_chat_service

    app.extensions["chat_service"] = build_chat_service(app.config)


def _configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.config["DEBUG"] else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app.logger.setLevel(level)


def _configure_extensions(app: Flask) -> None:
    CORS(app, resources={r"/api/*": {"origins": "*"}})


def _register_blueprints(app: Flask) -> None:
    from app.api.routes import api_bp
    from app.api.health import health_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(health_bp)


def _register_error_handlers(app: Flask) -> None:
    from app.api.errors import register_error_handlers

    register_error_handlers(app)


def _ensure_runtime_dirs(app: Flask) -> None:
    import os

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
