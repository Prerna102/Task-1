from fastapi import FastAPI

from app.db.database import Base, engine

from app.api.routes import health
from app.api.routes import ocr


# --------------------------------------------------
# Database
# --------------------------------------------------

# Create database tables
Base.metadata.create_all(
    bind=engine
)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="PaddleOCR API",
    description=(
        "OCR API using FastAPI, "
        "PaddleOCR and PostgreSQL"
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

app.include_router(
    ocr.router,
    prefix="/api/ocr",
    tags=["OCR"],
)


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "PaddleOCR API is running",
        "docs": "/docs",
        "health": "/health",
    }