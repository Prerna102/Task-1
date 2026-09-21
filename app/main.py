from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.database import Base, engine
from app.api.routes import health
from app.api.routes import ocr

from app.workers.ocr_worker import ocr_worker
import asyncio


# Database
# Create database tables
Base.metadata.create_all(
    bind=engine
)


#Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Start 2 background OCR workers
    worker_tasks = [
        asyncio.create_task(ocr_worker()),
        asyncio.create_task(ocr_worker()),
        asyncio.create_task(ocr_worker()),
        asyncio.create_task(ocr_worker())
    ]

    print("OCR workers started: 4")

    try:
        # Keep FastAPI running
        yield

    finally:
        # Stop workers when FastAPI shuts down
        print("Stopping OCR workers...")

        for task in worker_tasks:
            task.cancel()

        await asyncio.gather(
            *worker_tasks,
            return_exceptions=True,
        )

        print("OCR workers stopped.")


# FastAPI
app = FastAPI(
    title="PaddleOCR API",
    description=(
        "OCR API using FastAPI, "
        "PaddleOCR and PostgreSQL"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Routers
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


# Root
@app.get("/")
def root():
    return {
        "message": "PaddleOCR API is running",
        "docs": "/docs",
        "health": "/health",
    }
