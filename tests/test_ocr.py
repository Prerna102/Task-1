import os

# Use SQLite for tests instead of PostgreSQL
os.environ["DATABASE_URL"] = "sqlite:///./test.sqlite3"


from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.api.routes import ocr

-
# Test database


TEST_DATABASE_URL = "sqlite:///./test.sqlite3"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base.metadata.create_all(
    bind=engine
)


def override_get_db():

    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


# Mock PaddleOCR

def fake_process_document(image_path):

    return {
        "text": "Hello World",
        "confidence": 0.95,

        "details": [
            {
                "text": "Hello World",
                "confidence": 0.95,
                "box": [0, 0, 100, 50],
            }
        ],

        "tables": [],
        "layout": [],
        "parsing": [],

        "pages": [
            {
                "text": "Hello World",
                "confidence": 0.95,
                "details": [],
                "tables": [],
                "layout": [],
                "parsing": [],
            }
        ],
    }


# Replace real OCR with fake OCR
ocr.process_document = fake_process_document


# Test OCR upload


def test_ocr_upload():

    response = client.post(
        "/api/ocr",
        files={
            "file": (
                "test.jpg",
                b"fake image content",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True

    assert data["filename"] == "test.jpg"

    assert data["text"] == "Hello World"

    assert data["confidence"] == 0.95

    assert "id" in data

    assert "stored_filename" in data


def test_ocr_rejects_invalid_file_type():

    response = client.post(
        "/api/ocr",
        files={
            "file": (
                "test.exe",
                b"fake file",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert "Unsupported file type" in data["detail"]