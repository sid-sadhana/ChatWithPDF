"""Health-check endpoints used by container orchestrators."""
from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health() -> tuple:
    return jsonify(status="ok"), 200


@health_bp.get("/ready")
def ready() -> tuple:
    return jsonify(status="ready"), 200
