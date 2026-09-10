from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
import shutil

from app.database import Base, engine, get_db
from app.models import OCRResult
from app.ocr import extract_text

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="PaddleOCR FastAPI Service",
    description="OCR service using FastAPI, Uvicorn, PaddleOCR and PostgreSQL",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}


@app.get("/")
def root():
    return {
        "message": "PaddleOCR FastAPI service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an image and extract text using PaddleOCR.
    """

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}",
        )

    file_id = str(uuid.uuid4())
    filename = f"{file_id}{extension}"
    file_path = UPLOAD_DIR / filename

    try:
        # Save uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run OCR
        result = extract_text(str(file_path))

        # Save result in PostgreSQL
        ocr_record = OCRResult(
            filename=file.filename,
            stored_filename=filename,
            extracted_text=result["text"],
            confidence=result["confidence"],
        )

        db.add(ocr_record)
        db.commit()
        db.refresh(ocr_record)

        return {
            "id": ocr_record.id,
            "filename": file.filename,
            "text": result["text"],
            "confidence": result["confidence"],
        }

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(exc)}",
        )

    finally:
        file.file.close()


@app.get("/ocr/{ocr_id}")
def get_ocr_result(
    ocr_id: int,
    db: Session = Depends(get_db),
):
    """
    Get a previously stored OCR result.
    """

    result = (
        db.query(OCRResult)
        .filter(OCRResult.id == ocr_id)
        .first()
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail="OCR result not found",
        )

    return {
        "id": result.id,
        "filename": result.filename,
        "text": result.extracted_text,
        "confidence": result.confidence,
        "created_at": result.created_at,
    }


@app.get("/ocr")
def list_ocr_results(
    db: Session = Depends(get_db),
):
    """
    List all OCR results.
    """

    results = (
        db.query(OCRResult)
        .order_by(OCRResult.created_at.desc())
        .all()
    )

    return [
        {
            "id": result.id,
            "filename": result.filename,
            "text": result.extracted_text,
            "confidence": result.confidence,
            "created_at": result.created_at,
        }
        for result in results
    ]
