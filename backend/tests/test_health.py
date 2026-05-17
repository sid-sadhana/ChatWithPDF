from app import create_app
from app.config import TestingConfig


def test_health():
    app = create_app(TestingConfig)
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}
