import asyncio
import random
from pathlib import Path
from datetime import datetime


##recovery
async def recover_processing_jobs(
    db_session_factory,
    ocr_queue,
    ocr_model,
):
    """
    Recovers OCR jobs that were left in 'processing'
    after an application or worker restart.

    Jobs whose uploaded files still exist are placed
    back into the OCR queue.

    Jobs whose files no longer exist are marked failed.
    """

    db = db_session_factory()

    try:

        # Finds jobs that were interrupted while processing.
        records = (
            db.query(ocr_model)
            .filter(
                ocr_model.status == "processing"
            )
            .all()
        )

        print(
            f"Recovery found "
            f"{len(records)} unfinished job(s).",
            flush=True,
        )

        for record in records:

            # Check whether the uploaded file still exists.
            file_path = Path(
                record.file_path
            )

            if not file_path.exists():

                # File is gone, so this job cannot be recovered.
                record.status = "failed"

                record.error = (
                    "Uploaded file no longer exists."
                )

                record.completed_at = (
                    datetime.utcnow()
                )

                print(
                    f"Recovery failed: "
                    f"{record.filename} - "
                    f"file not found",
                    flush=True,
                )

                continue

            # Rebuild the worker job.
            job = {
                "job_id": record.job_id,
                "filename": record.filename,
                "file_path": record.file_path,
            }

            # Put the unfinished job back into the queue.
            await ocr_queue.put(job)

            # Change processing back to queued.
            record.status = "processing"

            record.error = None

            print(
                f"Recovered job: "
                f"{record.filename}",
                flush=True,
            )

        # Save database changes.
        db.commit()

    except Exception:

        # Roll back recovery changes if something fails.
        db.rollback()

        raise

    finally:

        # Close the database session.
        db.close()