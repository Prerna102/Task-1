# OCR POC

A simple OCR Proof of Concept using **FastAPI, PaddleOCR, Streamlit, and PostgreSQL**.  
It provides an API and web interface to upload images and extract text using PaddleOCR.

## Folder Structure

```text
Project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── streamlit_app.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       └── ocr.py
│   │
│   └── services/
│       ├── __init__.py
│       └── ocr_service.py
│
├── uploads/
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   └── test_ocr.py
│
├── .env
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Tech Stack

- **FastAPI** – Backend API
- **PaddleOCR** – OCR engine
- **Streamlit** – Web interface
- **PostgreSQL** – Database
- **SQLAlchemy** – Database ORM
- **Docker** – Containerization

## Run Locally

### 1. Create Virtual Environment

```bash
python -m venv env
```

### 2. Activate Environment

**Windows:**

```cmd
env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Database

For local testing without PostgreSQL:

```cmd
set DATABASE_URL=sqlite:///./sqlite.db
```

### 5. Start FastAPI

```cmd
set PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start Streamlit

Open another terminal and activate the environment:

```cmd
env\Scripts\activate
```

Set the API URL:

```cmd
set API_URL=http://127.0.0.1:8000
```

Start Streamlit:

```cmd
streamlit run app/streamlit_app.py
```

Streamlit:

```text
http://localhost:8501
```

## Run Using Docker

Make sure **Docker Desktop** is running.

### Build and Start

```bash
docker compose up --build
```

### Run in Background

```bash
docker compose up --build -d
```

### Stop Containers

```bash
docker compose down
```

### Services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## API

### Health Check

```http
GET /health
```

### OCR

```http
POST /api/ocr
```

Upload an image using the `file` parameter.

### Example Response

```json
{
  "success": true,
  "id": 1,
  "filename": "document.jpg",
  "text": "Extracted text...",
  "confidence": 0.95
}
```

## Notes

- This is a **Proof of Concept (POC)**.
- The lightweight PaddleOCR pipeline is focused on text extraction.
- Large images are resized before OCR to improve processing time.
- Table, layout, and document parsing features are not enabled in the lightweight OCR mode.