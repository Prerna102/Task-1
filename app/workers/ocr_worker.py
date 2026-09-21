import asyncio
from datetime import datetime

from app.db.database import SessionLocal
from app.db.models import OCRResult
from app.services.ocr_service import process_document
from app.utils.retry import retry_async

# Queue containing OCR jobs waiting to be processed.
ocr_queue = asyncio.Queue()


# Maximum number of OCR operations allowed simultaneously.
ocr_semaphore = asyncio.Semaphore(4)


def save_processing_result(
    job_id: str,
    result: dict,
):
    """Saves a successful OCR result to PostgreSQL."""

    # Creates a database session.
    db = SessionLocal()

    try:
        # Finds the corresponding OCR job.
        record = (
            db.query(OCRResult)
            .filter(OCRResult.job_id == job_id)
            .first()
        )

        # Makes sure the record exists.
        if record:

            # Stores extracted text.
            record.text = result["text"]

            # Stores OCR confidence.
            record.confidence = result["confidence"]

            # Marks processing as completed.
            record.status = "completed"

            # Stores completion time.
            record.completed_at = datetime.utcnow()

            # Saves changes.
            db.commit()

    finally:
        # Closes the database session.
        db.close()


def save_processing_error(
    job_id: str,
    error_message: str,
):
    """Saves a failed OCR job to PostgreSQL."""

    # Creates a database session.
    db = SessionLocal()

    try:
        # Finds the OCR job.
        record = (
            db.query(OCRResult)
            .filter(OCRResult.job_id == job_id)
            .first()
        )

        # Makes sure the record exists.
        if record:

            # Marks the job as failed.
            record.status = "failed"

            # Stores the error message.
            record.error = error_message

            # Stores completion time.
            record.completed_at = datetime.utcnow()

            # Saves changes.
            db.commit()

    finally:
        # Closes the database session.
        db.close()

async def process_ocr_job(job):
    """Runs OCR and saves the successful result."""

    # Runs blocking PaddleOCR in a separate thread.
    result = await asyncio.to_thread(
        process_document,
        job["file_path"],
    )

    # Saves the result without blocking the event loop.
    await asyncio.to_thread(
        save_processing_result,
        job["job_id"],
        result,
    )


async def ocr_worker():
    """Continuously processes OCR jobs from the queue."""

    while True:

        # Waits until a job becomes available.
        job = await ocr_queue.get()

        try:

            # Limits simultaneous OCR processing.
            async with ocr_semaphore:
                # Retries the OCR operation up to 3 times.
                await retry_async(
                    lambda: process_ocr_job(job),
                    attempts=3,
                    base_delay=1,
                )
                

                # Logs successful processing.
                print(
                    f"OCR completed: {job['filename']}"
                )

        except Exception as exc:

            # Converts the exception to text.
            error_message = str(exc)

            # Saves failure information to PostgreSQL.
            await asyncio.to_thread(
                save_processing_error,
                job["job_id"],
                error_message,
            )

            # Logs the failure.
            print(
                f"OCR failed: {job['filename']} - {error_message}"
            )

        finally:

            # Marks the queue job as completed.
            ocr_queue.task_done()
