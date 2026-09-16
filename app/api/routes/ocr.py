import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.db.database import SessionLocal
from app.db.models import OCRResult
from app.db.schemas import OCRStatusResponse, OCRUploadResponse
from app.workers.ocr_worker import ocr_queue


# Creates the OCR API router.
router = APIRouter()


# Directory where uploaded files are stored.
UPLOAD_DIR = Path("uploads")


# Creates the directory 
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Supported file extensions.
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".pdf",
}


@router.post(
    "/api/ocr",
    response_model=OCRUploadResponse,
)
async def upload_ocr(
    file: UploadFile = File(...),
):
    """Uploads a file and places it into the OCR queue."""

    # Checks whether a filename was provided.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    # Gets the uploaded file extension.
    extension = Path(
        file.filename
    ).suffix.lower()

    # Validates the file extension.
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    # Creates a unique job ID.
    job_id = str(uuid.uuid4())

    # Creates a unique physical filename.
    # #unique_filename = (
    #     f"{job_id}_{file.filename}"
    # )
    unique_filename = f"{job_id}{extension}"

    # Creates the destination path.
    file_path = UPLOAD_DIR / unique_filename

    try:

        # Opens the destination file.
        with open(file_path, "wb") as buffer:

            # Reads the uploaded file in chunks.
            while True:

                # Reads up to 1 MB.
                chunk = await file.read(1024 * 1024)

                # Stops when there is no more data.
                if not chunk:
                    break

                # Writes the chunk to disk.
                buffer.write(chunk)

    finally:

        # Closes the uploaded file.
        await file.close()

    # Creates a database session.
    db = SessionLocal()

    try:

        # Creates the initial OCR database record.
        record = OCRResult(
            job_id=job_id,
            filename=file.filename,
            file_path=str(file_path),
            status="processing",
        )

        # Adds the record.
        db.add(record)

        # Saves the record.
        db.commit()

    except Exception:

        # Rolls back the failed transaction.
        db.rollback()

        # Removes the uploaded file if DB creation failed.
        if file_path.exists():
            file_path.unlink()

        # Re-raises the error.
        raise

    finally:

        # Closes the database session.
        db.close()

    # Creates the background OCR job.
    job = {
        "job_id": job_id,
        "filename": file.filename,
        "file_path": str(file_path),
    }

    # Adds the job to the OCR queue.
    await ocr_queue.put(job)

    # Returns immediately without waiting for PaddleOCR.
    return {
        "success": True,
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
        "message": "File added to OCR queue",
    }


@router.get(
    "/api/ocr/status/{job_id}",
    response_model=OCRStatusResponse,
)
async def get_ocr_status(job_id: str):
    """Returns the current status/result of an OCR job."""

    # Creates a database session.
    db = SessionLocal()

    try:

        # Finds the requested job.
        record = (
            db.query(OCRResult)
            .filter(OCRResult.job_id == job_id)
            .first()
        )

        # Returns 404 if the job does not exist.
        if not record:
            raise HTTPException(
                status_code=404,
                detail="OCR job not found",
            )

        # Returns the current job information.
        return {
            "success": True,
            "job_id": record.job_id,
            "filename": record.filename,
            "status": record.status,
            "text": record.text,
            "confidence": record.confidence,
            "error": record.error,
        }

    finally:

        # Closes the database session.
        db.close()
