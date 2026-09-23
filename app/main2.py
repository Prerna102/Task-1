import os
import time
import asyncio
import psutil

from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.openapi.utils import get_openapi

from app.db.database import (
    Base,
    engine,
    SessionLocal,
)

from app.db.models import OCRResult

from app.api.routes import health
from app.api.routes import ocr

from app.workers.ocr_worker import (
    ocr_worker,
    ocr_queue,
)

from app.utils.recovery import (
    recover_processing_jobs,
)

from app.services.ocr_service import (
    process_document,
)


# Database

Base.metadata.create_all(
    bind=engine
)


# Benchmark settings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PDF_DIR = BASE_DIR / "test_pdfs2"
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}

NUMBER_OF_DOCS= 10

# Keep 2 benchmark workers for now.
WORKERS = 2


# Normal OCR workers
OCR_WORKERS = 2


@asynccontextmanager
async def lifespan(app: FastAPI):

    await recover_processing_jobs(
        SessionLocal,
        ocr_queue,
        OCRResult,
    )

    # Start normal background OCR workers.
    worker_tasks = [
        asyncio.create_task(
            ocr_worker()
        )
        for _ in range(OCR_WORKERS)
    ]

    print(
        f"OCR workers started: "
        f"{OCR_WORKERS}"
    )

    try:

        yield

    finally:

        print(
            "Stopping OCR workers..."
        )

        for task in worker_tasks:
            task.cancel()

        await asyncio.gather(
            *worker_tasks,
            return_exceptions=True,
        )

        print(
            "OCR workers stopped."
        )


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

app.openapi_version = "3.0.3"


# Custom OpenAPI

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    for component in schema.get(
        "components",
        {}
    ).get(
        "schemas",
        {}
    ).values():

        for prop in component.get(
            "properties",
            {}
        ).values():

            # Fix a single uploaded file.
            if (
                prop.get("contentMediaType")
                == "application/octet-stream"
            ):

                del prop["contentMediaType"]

                prop["format"] = "binary"

            # Fix multiple uploaded files.
            items = prop.get(
                "items",
                {}
            )

            if (
                items.get("contentMediaType")
                == "application/octet-stream"
            ):

                del items["contentMediaType"]

                items["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


app.openapi = custom_openapi


# Existing routers

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


# ---------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------

def process_pdf(pdf_path):
    """
    Process one PDF in one benchmark worker process.
    """

    pid = os.getpid()

    process = psutil.Process(pid)

    start_time = time.perf_counter()

    try:

        result = process_document(
            str(pdf_path)
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        # CPU used by this worker.
        cpu_usage = process.cpu_percent(
            interval=0.1
        )

        # RAM used by this worker.
        memory_mb = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        return {
            "pdf": pdf_path.name,
            "pid": pid,
            "time": elapsed,
            "cpu": cpu_usage,
            "memory": memory_mb,
            "success": True,
        }

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "pdf": pdf_path.name,
            "pid": pid,
            "time": elapsed,
            "cpu": 0,
            "memory": 0,
            "success": False,
            "error": str(e),
        }


def get_system_info():

    memory = psutil.virtual_memory()

    return {
        "physical_cpu_cores":
            psutil.cpu_count(
                logical=False
            ),

        "logical_cpu_cores":
            psutil.cpu_count(
                logical=True
            ),

        "total_ram_gb":
            round(
                memory.total
                / (1024 ** 3),
                2,
            ),

        "available_ram_gb":
            round(
                memory.available
                / (1024 ** 3),
                2,
            ),

        "used_ram_gb":
            round(
                memory.used
                / (1024 ** 3),
                2,
            ),

        "ram_usage_percent":
            memory.percent,

        "cpu_usage_percent":
            psutil.cpu_percent(
                interval=1
            ),
    }


# ---------------------------------------------------------
# System information endpoint
# ---------------------------------------------------------

@app.get(
    "/api/benchmark/system",
    tags=["Benchmark"],
)
def benchmark_system():

    return {
        "success": True,
        "system": get_system_info(),
        "benchmark_workers": WORKERS,
    }


# ---------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------

@app.post(
    "/api/ocr/test-batch",
    tags=["Benchmark"],
)
async def test_batch_ocr():

    # Check test_pdfs directory.
    if not TEST_PDF_DIR.exists():

        return {
            "success": False,
            "message": (
                "test_pdfs directory "
                "does not exist"
            ),
        }

    # Find PDFs.
    doc_files = sorted(
    file
    for file in TEST_PDF_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() in SUPPORTED_EXTENSIONS
)[:NUMBER_OF_DOCS]

    if not doc_files:

        return {
            "success": False,
            "message": (
                "No PDF files found "
                "in test_pdfs/"
            ),
        }

    # System information before OCR.
    before = psutil.virtual_memory()

    cpu_before = psutil.cpu_percent(
        interval=1
    )

    print()
    print("=" * 60)
    print("OCR BENCHMARK STARTED")
    print("=" * 60)

    print(
        f"PDFs selected : "
        f"{len(doc_files)}"
    )

    print(
        f"Workers       : "
        f"{WORKERS}"
    )

    print()

    print(
        f"Physical cores: "
        f"{psutil.cpu_count(logical=False)}"
    )

    print(
        f"Logical cores : "
        f"{psutil.cpu_count(logical=True)}"
    )

    print(
        f"Total RAM     : "
        f"{before.total / (1024 ** 3):.2f} GB"
    )

    print(
        f"RAM available : "
        f"{before.available / (1024 ** 3):.2f} GB"
    )

    print(
        f"RAM used      : "
        f"{before.used / (1024 ** 3):.2f} GB"
    )

    print(
        f"RAM usage     : "
        f"{before.percent:.1f}%"
    )

    print(
        f"CPU before    : "
        f"{cpu_before:.1f}%"
    )

    print()

    print("PDFs selected:")

    for pdf in doc_files:
        print(
            f"  {pdf.name}"
        )

    print()

    start_time = time.perf_counter()

    results = []

    # -----------------------------------------------------
    # ProcessPoolExecutor
    #
    # This creates exactly 2 benchmark worker processes.
    #
    # It also distributes pending PDFs between workers.
    # -----------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_pdf,
                pdf,
            ): pdf
            for pdf in doc_files
        }

        for future in as_completed(
            futures
        ):

            result = future.result()

            results.append(
                result
            )

            if result["success"]:

                print(
                    f"PID {result['pid']} | "
                    f"{result['pdf']} | "
                    f"{result['time']:.2f}s | "
                    f"CPU {result['cpu']:.1f}% | "
                    f"RAM "
                    f"{result['memory']:.0f} MB"
                )

            else:

                print(
                    f"PID {result['pid']} | "
                    f"{result['pdf']} | "
                    f"FAILED | "
                    f"{result['error']}"
                )

    total_time = (
        time.perf_counter()
        - start_time
    )

    # System information after OCR.
    after = psutil.virtual_memory()

    cpu_after = psutil.cpu_percent(
        interval=1
    )

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    failed = (
        len(results)
        - successful
    )

    worker_pids = sorted(
        set(
            result["pid"]
            for result in results
        )
    )

    successful_memory = [
        result["memory"]
        for result in results
        if result["success"]
    ]

    if successful_memory:

        average_memory = (
            sum(successful_memory)
            / len(successful_memory)
        )

        maximum_memory = max(
            successful_memory
        )

    else:

        average_memory = 0

        maximum_memory = 0

    throughput = (
        len(doc_files)
        / total_time
    )

    print()
    print("=" * 60)
    print("FINAL BENCHMARK RESULT")
    print("=" * 60)

    print(
        f"Workers            : "
        f"{WORKERS}"
    )

    print(
        f"Worker PIDs        : "
        f"{worker_pids}"
    )

    print(
        f"PDFs processed     : "
        f"{len(results)}"
    )

    print(
        f"Successful         : "
        f"{successful}"
    )

    print(
        f"Failed             : "
        f"{failed}"
    )

    print(
        f"Total time         : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Throughput         : "
        f"{throughput:.3f} PDFs/sec"
    )

    print(
        f"Average worker RAM : "
        f"{average_memory:.0f} MB"
    )

    print(
        f"Maximum worker RAM : "
        f"{maximum_memory:.0f} MB"
    )

    print()

    print(
        "SYSTEM AFTER OCR"
    )

    print("-" * 60)

    print(
        f"CPU usage          : "
        f"{cpu_after:.1f}%"
    )

    print(
        f"RAM usage          : "
        f"{after.percent:.1f}%"
    )

    print(
        f"RAM used           : "
        f"{after.used / (1024 ** 3):.2f} GB"
    )

    print(
        f"RAM available      : "
        f"{after.available / (1024 ** 3):.2f} GB"
    )

    # Return everything as JSON as well.
    return {
        "success": True,

        "benchmark": {
            "workers": WORKERS,
            "pdf_count": len(doc_files),
            "total_time_seconds":
                round(
                    total_time,
                    2,
                ),
            "throughput_pdfs_per_second":
                round(
                    throughput,
                    3,
                ),
        },

        "workers": {
            "pids": worker_pids,
            "average_ram_mb":
                round(
                    average_memory,
                    2,
                ),
            "maximum_ram_mb":
                round(
                    maximum_memory,
                    2,
                ),
        },

        "system_before": {
            "cpu_percent":
                cpu_before,
            "ram_percent":
                before.percent,
            "ram_used_gb":
                round(
                    before.used
                    / (1024 ** 3),
                    2,
                ),
            "ram_available_gb":
                round(
                    before.available
                    / (1024 ** 3),
                    2,
                ),
        },

        "system_after": {
            "cpu_percent":
                cpu_after,
            "ram_percent":
                after.percent,
            "ram_used_gb":
                round(
                    after.used
                    / (1024 ** 3),
                    2,
                ),
            "ram_available_gb":
                round(
                    after.available
                    / (1024 ** 3),
                    2,
                ),
        },

        "files": results,
    }


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": (
            "PaddleOCR API is running"
        ),
        "docs": "/docs",
        "health": "/health",
        "test_batch":
            "/api/ocr/test-batch",
        "benchmark_system":
            "/api/benchmark/system",
    }
