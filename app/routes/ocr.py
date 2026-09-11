from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Depends,
)

from sqlalchemy.orm import Session

from pathlib import Path
import uuid
import shutil


from app.db.database import get_db
from app.db.models import OCRResult

from app.services.ocr_service import (
    process_document,
)


# --------------------------------------------------
# Router
# --------------------------------------------------

router = APIRouter()


# --------------------------------------------------
# Upload directory
# --------------------------------------------------

UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# POST /api/ocr
# --------------------------------------------------

@router.post("")
async def perform_ocr(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an image/document, process it with
    PaddleOCR and save the OCR result to PostgreSQL.
    """

    # --------------------------------------------------
    # Validate filename
    # --------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )


    # --------------------------------------------------
    # Validate extension
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Generate unique filename
    # --------------------------------------------------

    file_id = str(
        uuid.uuid4()
    )


    stored_filename = (
        f"{file_id}{extension}"
    )


    file_path = (
        UPLOAD_DIR / stored_filename
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
        # Run PaddleOCR
        # --------------------------------------------------

        ocr_result = process_document(
            str(file_path)
        )


        # --------------------------------------------------
        # Save result to PostgreSQL
        # --------------------------------------------------

        record = OCRResult(
            filename=file.filename,
            stored_filename=stored_filename,
            extracted_text=ocr_result.get(
                "text",
                "",
            ),
            confidence=ocr_result.get(
                "confidence",
                0.0,
            ),
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

            "stored_filename": stored_filename,

            "text": ocr_result.get(
                "text",
                "",
            ),

            "confidence": ocr_result.get(
                "confidence",
                0.0,
            ),

            "details": ocr_result.get(
                "details",
                [],
            ),

            "tables": ocr_result.get(
                "tables",
                [],
            ),

            "layout": ocr_result.get(
                "layout",
                [],
            ),

            "parsing": ocr_result.get(
                "parsing",
                [],
            ),

            "pages": ocr_result.get(
                "pages",
                [],
            ),
        }


    except Exception as exc:

        # Rollback database transaction
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
# GET /api/ocr/{ocr_id}
# --------------------------------------------------

@router.get("/{ocr_id}")
def get_ocr_result(
    ocr_id: int,
    db: Session = Depends(get_db),
):
    """
    Get one OCR result by ID.
    """

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
# GET /api/ocr
# --------------------------------------------------

@router.get("")
def list_ocr_results(
    db: Session = Depends(get_db),
):
    """
    Return all OCR results.
    """

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