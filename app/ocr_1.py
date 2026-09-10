from pathlib import Path
from typing import Any
import json
import re

from paddleocr import PPStructureV3


# 1. INITIALIZE PP-STRUCTURE-V3 ONCE


pipeline = PPStructureV3(
    lang="en",

    # -----------------------------
    # Document preprocessing
    # -----------------------------
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,

    # -----------------------------
    # Text line orientation
    # -----------------------------
    use_textline_orientation=True,
)


# 2. HELPER


def _result_to_dict(result: Any) -> dict:
    """
    Convert PaddleOCR Result object to Python dictionary.
    """

    data = result.json

    if isinstance(data, str):
        data = json.loads(data)

    # PPStructureV3 returns:
    #
    # {
    #     "res": {
    #         ...
    #     }
    # }
    #
    # So return the contents of "res".

    return data.get("res", data)


# 3. TEXT CLEANING


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


# 4. EXTRACT NORMAL TEXT
# ============================================================

def extract_text_from_result(data: dict) -> dict:
    """
    Extract normal OCR text from PPStructureV3 result.

    PPStructureV3 provides global OCR results in:

        overall_ocr_res

    Important fields:

        rec_texts
        rec_scores
        rec_polys
        rec_boxes
    """

    ocr_result = data.get(
        "overall_ocr_res",
        {}
    )

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

        # ----------------------------------------
        # Confidence
        # ----------------------------------------

        score = 0.0

        if index < len(scores):

            try:
                score = float(
                    scores[index]
                )

            except (
                TypeError,
                ValueError
            ):
                score = 0.0

        # ----------------------------------------
        # Bounding box
        # ----------------------------------------

        box = None

        if index < len(boxes):
            box = boxes[index]

        # ----------------------------------------
        # Store text
        # ----------------------------------------

        extracted_text.append(text)

        detailed_text.append(
            {
                "text": text,
                "confidence": round(
                    score,
                    4
                ),
                "box": box,
            }
        )

        valid_scores.append(score)

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
        "text": clean_text(text),

        "confidence": round(
            average_confidence,
            4
        ),

        "details": detailed_text,
    }


# ============================================================
# 5. EXTRACT DOCUMENT LAYOUT
# ============================================================

def extract_layout(data: dict) -> list:
    """
    Extract document layout information.

    PPStructureV3 can identify regions such as:

        text
        table
        image
        paragraph_title
        doc_title
        etc.

    """

    layout_result = data.get(
        "layout_det_res",
        {}
    )

    boxes = layout_result.get(
        "boxes",
        []
    )

    layout = []

    for box in boxes:

        layout.append(
            {
                "label": box.get(
                    "label"
                ),

                "confidence": round(
                    float(
                        box.get(
                            "score",
                            0.0
                        )
                    ),
                    4
                ),

                "bbox": box.get(
                    "coordinate"
                ),
            }
        )

    return layout


# ============================================================
# 6. EXTRACT TABLES
# ============================================================

def extract_tables(data: dict) -> list:
    """
    Extract tables detected by PPStructureV3.

    PPStructureV3 handles:

        table detection
        table structure recognition
        cell recognition
        OCR inside table

    We DO NOT manually align rows/columns here.

    PaddleOCR provides:

        pred_html
        cell_box_list
        table_ocr_pred

    """

    table_results = data.get(
        "table_res_list",
        []
    )

    tables = []

    for table_index, table in enumerate(
        table_results
    ):

        # ----------------------------------------------------
        # Table bounding boxes
        # ----------------------------------------------------

        cell_boxes = table.get(
            "cell_box_list",
            []
        )

        # ----------------------------------------------------
        # HTML representation
        # ----------------------------------------------------

        html = table.get(
            "pred_html",
            ""
        )

        # ----------------------------------------------------
        # OCR results inside cells
        # ----------------------------------------------------

        table_ocr = table.get(
            "table_ocr_pred",
            {}
        )

        cell_texts = table_ocr.get(
            "rec_texts",
            []
        )

        cell_scores = table_ocr.get(
            "rec_scores",
            []
        )

        cell_boxes_from_ocr = table_ocr.get(
            "rec_boxes",
            []
        )

        # ----------------------------------------------------
        # Build cell information
        # ----------------------------------------------------

        cells = []

        for index, text in enumerate(
            cell_texts
        ):

            if not text:
                continue

            score = 0.0

            if index < len(cell_scores):

                try:
                    score = float(
                        cell_scores[index]
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    score = 0.0

            box = None

            if index < len(cell_boxes_from_ocr):

                box = cell_boxes_from_ocr[
                    index
                ]

            elif index < len(cell_boxes):

                box = cell_boxes[
                    index
                ]

            cells.append(
                {
                    "text": clean_text(
                        text
                    ),

                    "confidence": round(
                        score,
                        4
                    ),

                    "bbox": box,
                }
            )

        # ----------------------------------------------------
        # Average table confidence
        # ----------------------------------------------------

        table_scores = [
            cell["confidence"]
            for cell in cells
        ]

        table_confidence = (
            sum(table_scores)
            / len(table_scores)
            if table_scores
            else 0.0
        )

        # ----------------------------------------------------
        # Store complete table
        # ----------------------------------------------------

        tables.append(
            {
                "table_id": table_index,

                # PaddleOCR's recognized
                # table structure
                "html": html,

                # OCR text detected inside cells
                "cells": cells,

                "confidence": round(
                    table_confidence,
                    4
                ),
            }
        )

    return tables


# ============================================================
# 7. EXTRACT PARSING RESULTS
# ============================================================

def extract_parsing_results(data: dict) -> list:
    """
    Extract PaddleOCR's reading-order/layout parsing results.

    Each parsing block can contain:

        block_label
        block_bbox
        block_content
        block_order
    """

    parsing_results = data.get(
        "parsing_res_list",
        []
    )

    blocks = []

    for block in parsing_results:

        blocks.append(
            {
                "label": block.get(
                    "block_label"
                ),

                "bbox": block.get(
                    "block_bbox"
                ),

                "content": block.get(
                    "block_content",
                    ""
                ),

                "order": block.get(
                    "block_order"
                ),
            }
        )

    return blocks


# ============================================================
# 8. PROCESS ONE PAGE
# ============================================================

def process_page(result: Any) -> dict:
    """
    Process one PPStructureV3 prediction result.
    """

    data = _result_to_dict(
        result
    )

    # Normal text
   

    text_result = extract_text_from_result(
        data
    )

    # Document layout
    

    layout = extract_layout(
        data
    )

    # Tables
    

    tables = extract_tables(
        data
    )

    # Reading-order parsing
  

    parsing = extract_parsing_results(
        data
    )

    return {
        "text": text_result["text"],

        "confidence": text_result[
            "confidence"
        ],

        "details": text_result[
            "details"
        ],

        "layout": layout,

        "tables": tables,

        "parsing": parsing,
    }


# 9. COMPLETE OCR PIPELINE


def process_document(image_path: str) -> dict:
   

    
    # STEP 1 — VALIDATE INPUT
  

    path = Path(
        image_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Path is not a file: {image_path}"
        )

    # STEP 2 — RUN PPSTRUCTUREV3


    results = pipeline.predict(
        input=image_path
    )

    # STEP 3 — PROCESS RESULTS
   

    pages = []

    for result in results:

        page_result = process_page(
            result
        )

        pages.append(
            page_result
        )

    # STEP 4 — COMBINE TEXT
    

    all_text = []

    all_confidences = []

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

        all_tables.extend(
            page["tables"]
        )

        all_layout.extend(
            page["layout"]
        )

        all_parsing.extend(
            page["parsing"]
        )

    # STEP 5 — DOCUMENT CONFIDENCE
    

    confidence = (
        sum(all_confidences)
        / len(all_confidences)
        if all_confidences
        else 0.0
    )

    # STEP 6 — FINAL RESULT
    

    return {
        "text": clean_text(
            "\n".join(all_text)
        ),

        "confidence": round(
            confidence,
            4
        ),

        "tables": all_tables,

        "layout": all_layout,

        "parsing": all_parsing,

        "pages": pages,
    }