"""Centralized error handlers."""
from __future__ import annotations

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from app.services.exceptions import ServiceError


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ServiceError)
    def _service_error(err: ServiceError):
        return jsonify(error=str(err)), err.status_code

    @app.errorhandler(HTTPException)
    def _http_error(err: HTTPException):
        return jsonify(error=err.description), err.code or 500

    @app.errorhandler(Exception)
    def _unhandled(err: Exception):
        app.logger.exception("Unhandled exception: %s", err)
        return jsonify(error="Internal server error"), 500
