from pathlib import Path
from typing import Any
import json
import re
from PIL import Image
import tempfile
from paddleocr import PaddleOCR


# ============================================================
# 1. INITIALIZE LIGHTWEIGHT OCR PIPELINE ONCE
# ============================================================

ocr = PaddleOCR(
    lang="en",
)


# ============================================================
# 2. HELPERS
# ============================================================

def _result_to_dict(result: Any) -> dict:
    """
    Convert PaddleOCR Result object to a Python dictionary.
    """

    try:
        data = result.json
    except Exception:
        data = result

    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        return {}

    return data.get("res", data)


# ============================================================
# 3. TEXT CLEANING
# ============================================================

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

        # Replace multiple spaces with one
        line = re.sub(r"\s+", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# 4. EXTRACT NORMAL OCR TEXT
# ============================================================

def extract_text_from_result(data: dict) -> dict:
    """
    Extract normal OCR text from PaddleOCR result.

    Compatible with the existing FastAPI response:
        text
        confidence
        details
    """

    # PaddleOCR normally returns OCR results under rec_texts,
    # rec_scores and rec_boxes.

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])

    # Some versions may place them under OCR result.
    if not texts:
        ocr_result = data.get("ocr_res", {})

        if isinstance(ocr_result, dict):
            texts = ocr_result.get("rec_texts", [])
            scores = ocr_result.get("rec_scores", [])
            boxes = ocr_result.get("rec_boxes", [])

    extracted_text = []
    detailed_text = []
    valid_scores = []

    for index, text in enumerate(texts):

        if not text:
            continue

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        score = 0.0

        if index < len(scores):

            try:
                score = float(scores[index])

            except (TypeError, ValueError):
                score = 0.0

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        box = None

        if index < len(boxes):
            box = boxes[index]

        # ----------------------------------------------------
        # Store text
        # ----------------------------------------------------

        extracted_text.append(str(text))

        detailed_text.append(
            {
                "text": str(text),
                "confidence": round(score, 4),
                "box": box,
            }
        )

        valid_scores.append(score)

    text = "\n".join(extracted_text)

    average_confidence = (
        sum(valid_scores) / len(valid_scores)
        if valid_scores
        else 0.0
    )

    return {
        "text": clean_text(text),
        "confidence": round(average_confidence, 4),
        "details": detailed_text,
    }


# ============================================================
# 5. EMPTY COMPATIBILITY HELPERS
# ============================================================

def extract_layout(data: dict) -> list:
    """
    Lightweight OCR does not perform document layout detection.

    Kept for API compatibility with the previous PPStructureV3
    implementation.
    """

    return []


def extract_tables(data: dict) -> list:
    """
    Lightweight OCR does not perform table structure recognition.

    Kept for API compatibility with the previous PPStructureV3
    implementation.
    """

    return []


def extract_parsing_results(data: dict) -> list:
    """
    Lightweight OCR does not perform document parsing or
    reading-order analysis.

    Kept for API compatibility with the previous PPStructureV3
    implementation.
    """

    return []


# ============================================================
# 6. PROCESS ONE PAGE / IMAGE
# ============================================================

def process_page(result: Any) -> dict:
    """
    Process one PaddleOCR prediction result.

    Returns the same response structure expected by
    the existing FastAPI application.
    """

    data = _result_to_dict(result)

    # --------------------------------------------------------
    # Normal OCR text
    # --------------------------------------------------------

    text_result = extract_text_from_result(data)

    # --------------------------------------------------------
    # Lightweight OCR does not calculate these
    # --------------------------------------------------------

    layout = extract_layout(data)
    tables = extract_tables(data)
    parsing = extract_parsing_results(data)

    return {
        "text": text_result["text"],
        "confidence": text_result["confidence"],
        "details": text_result["details"],
        "layout": layout,
        "tables": tables,
        "parsing": parsing,
    }


# ============================================================
# 7. COMPLETE OCR PIPELINE
# ============================================================


def process_document(image_path: str) -> dict:
    """
    Run lightweight PaddleOCR on an image/document.

    Compatible with the existing FastAPI route.

    Returns:

    {
        "text": "...",
        "confidence": 0.95,
        "details": [...],
        "tables": [],
        "layout": [],
        "parsing": [],
        "pages": [...]
    }
    """

    # ========================================================
    # STEP 1 — VALIDATE INPUT
    # ========================================================

    path = Path(image_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    # ========================================================
    # STEP 2 — RUN LIGHTWEIGHT PADDLEOCR
    # ========================================================

    try:

        results = ocr.predict(
            image_path
        )

    except Exception as exc:

        raise RuntimeError(
            f"PaddleOCR inference failed: {exc}"
        ) from exc

    # ========================================================
    # STEP 3 — PROCESS RESULTS
    # ========================================================

    pages = []

    for result in results:

        page_result = process_page(result)

        pages.append(page_result)

    # ========================================================
    # STEP 4 — COMBINE RESULTS
    # ========================================================

    all_text = []
    all_confidences = []
    all_details = []

    all_tables = []
    all_layout = []
    all_parsing = []

    for page in pages:

        if page["text"]:
            all_text.append(page["text"])

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

    # ========================================================
    # STEP 5 — DOCUMENT CONFIDENCE
    # ========================================================

    confidence = (
        sum(all_confidences) / len(all_confidences)
        if all_confidences
        else 0.0
    )

    # ========================================================
    # STEP 6 — FINAL RESULT
    # ========================================================

    return {
        "text": clean_text(
            "\n".join(all_text)
        ),

        "confidence": round(
            confidence,
            4
        ),

        "details": all_details,

        # Kept for FastAPI compatibility.
        # Lightweight OCR does not generate these.
        "tables": all_tables,
        "layout": all_layout,
        "parsing": all_parsing,

        "pages": pages,
    }
