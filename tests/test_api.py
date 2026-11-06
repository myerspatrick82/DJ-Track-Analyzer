from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_invalid_file_type():
    response = client.post(
        "/analyze",
        files ={
            "file": ("test.txt", b"not audio", "text/plain")
        }
    )

    assert response.status_code == 400

def test_file_too_large():
    response = client.post(
        "/analyze",
        files = {
            "file": (
                "large.wav",
                b"0" * (25 * 1024 * 1024 + 1),
                "audio/wav"
            )
        }
    )

    assert response.status_code == 413