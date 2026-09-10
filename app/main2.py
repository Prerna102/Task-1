from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
)

from sqlalchemy.orm import Session

from pathlib import Path

import uuid
import shutil


from app.database import (
    Base,
    engine,
    get_db,
)

from app.models import OCRResult

from app.ocr import extract_text


# --------------------------------------------------
# Database
# --------------------------------------------------

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
# Upload directory
# --------------------------------------------------

UPLOAD_DIR = Path(
    "uploads"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# Allowed file extensions
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".pdf",
}


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "PaddleOCR API is running",
        "docs": "/docs",
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# OCR endpoint
# --------------------------------------------------

@app.post("/api/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )


    extension = Path(
        file.filename
    ).suffix.lower()


    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Supported formats: "
                "JPG, JPEG, PNG, BMP, WEBP, "
                "TIFF and PDF."
            ),
        )


    # Generate unique filename
    file_id = str(
        uuid.uuid4()
    )


    stored_filename = (
        f"{file_id}{extension}"
    )


    file_path = (
        UPLOAD_DIR /
        stored_filename
    )


    try:

        # --------------------------------------------------
        # Save uploaded file
        # --------------------------------------------------

        with file_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )


        # --------------------------------------------------
        # Run OCR
        # --------------------------------------------------

        ocr_result = extract_text(
            str(file_path)
        )


        # --------------------------------------------------
        # Save result to PostgreSQL
        # --------------------------------------------------

        record = OCRResult(
            filename=file.filename,
            stored_filename=stored_filename,
            extracted_text=ocr_result["text"],
            confidence=ocr_result["confidence"],
        )


        db.add(record)

        db.commit()

        db.refresh(record)


        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        return {
            "success": True,
            "id": record.id,
            "filename": file.filename,
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
        }


    except Exception as exc:

        db.rollback()


        raise HTTPException(
            status_code=500,
            detail=(
                f"OCR processing failed: {str(exc)}"
            ),
        )


    finally:

        file.file.close()


# --------------------------------------------------
# Get OCR result
# --------------------------------------------------

@app.get("/api/ocr/{ocr_id}")
def get_ocr_result(
    ocr_id: int,
    db: Session = Depends(get_db),
):

    result = (
        db.query(OCRResult)
        .filter(
            OCRResult.id == ocr_id
        )
        .first()
    )


    if not result:

        raise HTTPException(
            status_code=404,
            detail="OCR result not found.",
        )


    return {
        "id": result.id,
        "filename": result.filename,
        "text": result.extracted_text,
        "confidence": result.confidence,
        "created_at": result.created_at,
    }


# --------------------------------------------------
# List OCR results
# --------------------------------------------------

@app.get("/api/ocr")
def list_ocr_results(
    db: Session = Depends(get_db),
):

    results = (
        db.query(OCRResult)
        .order_by(
            OCRResult.created_at.desc()
        )
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
