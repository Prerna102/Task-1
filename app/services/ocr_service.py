from pathlib import Path
from typing import Any
import asyncio
import json
import re
import threading
import gc
#import fitz
import pymupdf
#from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
# 1. CONFIGURATION

PDF_WORKERS = 4
PDF_SEMAPHORE = 4
PDF_DPI = 150


# 2. THREAD-LOCAL OCR

_ocr_local = threading.local()


def get_ocr() -> PaddleOCR:
    """
    Return one PaddleOCR instance for the current worker thread.

    Each thread gets its own OCR instance.
    """

    if not hasattr(_ocr_local, "ocr"):

        _ocr_local.ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    return _ocr_local.ocr


# 3. HELPERS


def _result_to_dict(result: Any) -> dict:
    """
    Convert PaddleOCR Result object to a Python dictionary.
    """

    try:
        data = result.json
    except Exception:
        data = result

    if callable(data):
        data = data()

    if isinstance(data, str):

        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return {}

    if not isinstance(data, dict):
        return {}

    return data.get("res", data)


# 4. TEXT CLEANING


def clean_text(text: str) -> str:
    """
    Clean OCR text without changing its meaning.
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Replace multiple spaces with one.
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# 5. EXTRACT NORMAL OCR TEXT


def extract_text_from_result(data: dict) -> dict:
    """
    Extract normal OCR text from PaddleOCR result.

    Returns:
        text
        confidence
        details
    """

    texts = data.get(
        "rec_texts",
        []
    )

    scores = data.get(
        "rec_scores",
        []
    )

    boxes = data.get(
        "rec_boxes",
        []
    )

    # Some PaddleOCR versions may place OCR
    # information inside ocr_res.

    if not texts:

        ocr_result = data.get(
            "ocr_res",
            {}
        )

        if isinstance(ocr_result, dict):

            texts = ocr_result.get(
                "rec_texts",
                []
            )

            scores = ocr_result.get(
                "rec_scores",
                []
            )

            boxes = ocr_result.get(
                "rec_boxes",
                []
            )

    extracted_text = []
    detailed_text = []
    valid_scores = []

    for index, text in enumerate(texts):

        if not text:
            continue

        # Extract confidence score.

        score = 0.0

        if index < len(scores):

            try:
                score = float(
                    scores[index]
                )

            except (TypeError, ValueError):
                score = 0.0

        # Extract bounding box.

        box = None

        if index < len(boxes):
            box = boxes[index]

        # Store detected text.

        extracted_text.append(
            str(text)
        )

        detailed_text.append(
            {
                "text": str(text),
                "confidence": round(
                    score,
                    4
                ),
                "box": box,
            }
        )

        valid_scores.append(
            score
        )

    text = "\n".join(
        extracted_text
    )

    average_confidence = (
        sum(valid_scores)
        / len(valid_scores)
        if valid_scores
        else 0.0
    )

    return {
        "text": clean_text(
            text
        ),
        "confidence": round(
            average_confidence,
            4
        ),
        "details": detailed_text,
    }


# 6. EMPTY COMPATIBILITY HELPERS


def extract_layout(data: dict) -> list:
    """
    Lightweight OCR does not perform document layout detection.

    Kept for API compatibility.
    """

    return []


def extract_tables(data: dict) -> list:
    """
    Lightweight OCR does not perform table structure recognition.

    Kept for API compatibility.
    """

    return []


def extract_parsing_results(data: dict) -> list:
    """
    Lightweight OCR does not perform document parsing
    or reading-order analysis.

    Kept for API compatibility.
    """

    return []


# 7. PROCESS ONE PADDLEOCR RESULT


def process_page(result: Any) -> dict:
    """
    Process one PaddleOCR prediction result.
    """

    data = _result_to_dict(
        result
    )
    print("PADDLE DATA KEYS:", data.keys())
    print("REC TEXTS:", data.get("rec_texts"))
    print("REC SCORES:", data.get("rec_scores"))

    text_result = extract_text_from_result(
        data
    )

    return {
        "text": text_result["text"],
        "confidence": text_result["confidence"],
        "details": text_result["details"],
        "layout": extract_layout(data),
        "tables": extract_tables(data),
        "parsing": extract_parsing_results(data),
    }


# 8. OCR ONE PDF PAGE

def process_pdf_page_sync(
    pdf_path: str,
    page_number: int,
) -> dict:
    """
    Convert one PDF page to a PIL image,
    run PaddleOCR, then release the image.
    """

    image = None
    pix = None

    try:
        # Open the PDF and select the requested page.
        with pymupdf.open(pdf_path) as pdf:
            page = pdf[page_number - 1]

            # Render the PDF page at approximately 144 DPI.
            matrix = pymupdf.Matrix(2, 2)

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            # Convert PyMuPDF pixels into a PIL image.
            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )

            image_array = np.array(image)

            

        print(
            f"PAGE {page_number} IMAGE SIZE: {image_array.size}",
            flush=True,
        )

        # Get the OCR instance belonging to the current worker thread.
        ocr = get_ocr()

        # Run PaddleOCR on the PIL image.
        results = ocr.predict(image_array)

        print(
            f"PAGE {page_number} RAW RESULTS: {results}",
            flush=True,
        )

        page_text = []
        page_details = []
        page_confidences = []

        for result in results:
            processed = process_page(result)

            if processed["text"]:
                page_text.append(processed["text"])

            page_details.extend(
                processed["details"]
            )

            page_confidences.append(
                processed["confidence"]
            )

        page_confidence = (
            sum(page_confidences)
            / len(page_confidences)
            if page_confidences
            else 0.0
        )

        return {
            "page_number": page_number,
            "text": clean_text(
                "\n".join(page_text)
            ),
            "confidence": round(
                page_confidence,
                4,
            ),
            "details": page_details,
            "layout": [],
            "tables": [],
            "parsing": [],
        }

    except Exception as exc:
        print(
            f"OCR ERROR ON PAGE {page_number}: {exc}",
            flush=True,
        )

        return {
            "page_number": page_number,
            "text": "",
            "confidence": 0.0,
            "details": [],
            "layout": [],
            "tables": [],
            "parsing": [],
            "error": str(exc),
        }

    finally:
        # Release the PIL image.
        if image is not None:
            try:
                image.close()
            except Exception:
                pass
            del image

        # Release the PyMuPDF pixmap.
        pix = None

        # Clean unused objects.
        gc.collect()




# 9. ASYNC PDF PROCESSING

async def process_pdf(
    pdf_path: str
) -> dict:
    """
    Process a multipage PDF using four concurrent
    OCR worker threads.

    Semaphore ensures that no more than four pages
    are processed at the same time.
    """

    path = Path(
        pdf_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Use PyMuPDF only to determine the number of pages.
    with pymupdf.open(pdf_path) as pdf:

        total_pages = len(pdf)

    if total_pages == 0:

        return {
            "text": "",
            "confidence": 0.0,
            "details": [],
            "tables": [],
            "layout": [],
            "parsing": [],
            "pages": [],
        }

    # Maximum of four page tasks can enter
    # the OCR section at the same time.
    semaphore = asyncio.Semaphore(
        PDF_SEMAPHORE
    )

    async def process_one_page(
        page_number: int
    ) -> dict:

        async with semaphore:

            # PDF conversion and PaddleOCR are blocking,
            # so run them in a thread.
            return await asyncio.to_thread(
                process_pdf_page_sync,
                pdf_path,
                page_number,
            )

    # Create one async task for every PDF page.
    tasks = [
        process_one_page(
            page_number
        )
        for page_number in range(
            1,
            total_pages + 1
        )
    ]

    # Wait for all pages to finish.
    pages = await asyncio.gather(
        *tasks
    )

    # Restore original PDF page order.
    pages.sort(
        key=lambda page:
        page["page_number"]
    )

    # Combine page results.

    all_text = []
    all_details = []
    all_confidences = []

    for page in pages:

        if page["text"]:

            all_text.append(
                f"Page {page['page_number']}\n"
                f"{page['text']}"
            )

        all_details.extend(
            page["details"]
        )

        all_confidences.append(
            page["confidence"]
        )

    document_confidence = (
        sum(all_confidences)
        / len(all_confidences)
        if all_confidences
        else 0.0
    )

    return {
        "text": clean_text(
            "\n\n".join(
                all_text
            )
        ),

        "confidence": round(
            document_confidence,
            4
        ),

        "details": all_details,

        "tables": [],

        "layout": [],

        "parsing": [],

        "pages": pages,
    }


# 10. PROCESS NORMAL IMAGE

def process_image(
    image_path: str
) -> dict:
    """
    Run PaddleOCR directly on an image.
    """

    try:

        ocr = get_ocr()

        results = ocr.predict(
            image_path
        )
      
    except Exception as exc:

        raise RuntimeError(
            f"PaddleOCR inference failed: {exc}"
        ) from exc

    pages = []

    for result in results:

        page_result = process_page(
            result
        )

        pages.append(
            page_result
        )

    all_text = []
    all_confidences = []
    all_details = []

    all_tables = []
    all_layout = []
    all_parsing = []

    for page in pages:

        if page["text"]:

            all_text.append(
                page["text"]
            )

        all_confidences.append(
            page["confidence"]
        )

        all_details.extend(
            page["details"]
        )

        all_tables.extend(
            page["tables"]
        )

        all_layout.extend(
            page["layout"]
        )

        all_parsing.extend(
            page["parsing"]
        )

    confidence = (
        sum(all_confidences)
        / len(all_confidences)
        if all_confidences
        else 0.0
    )

    return {
        "text": clean_text(
            "\n".join(all_text)
        ),

        "confidence": round(
            confidence,
            4
        ),

        "details": all_details,

        "tables": all_tables,
        "layout": all_layout,
        "parsing": all_parsing,

        "pages": pages,
    }


# 11. COMPLETE OCR PIPELINE

def process_document(
    image_path: str
) -> dict:
    """
    Main OCR entry point.

    Images:
        Image → PaddleOCR

    PDFs:
        PDF → pdf2image → page images
            → four concurrent OCR workers
            → combine results
    """

    # Validate input.

    path = Path(
        image_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {image_path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    # PDF processing.

    if path.suffix.lower() == ".pdf":

        return asyncio.run(
            process_pdf(
                image_path
            )
        )

    # Normal image processing.

    return process_image(
        image_path
    )

