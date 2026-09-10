from paddleocr import PaddleOCR


# Initialize PaddleOCR once when the application starts.
# This avoids loading the OCR model for every request.
ocr = PaddleOCR(
    lang="en",
)


def extract_text(image_path: str) -> dict:
    """
    Run PaddleOCR on an image and return extracted text
    and average confidence.
    """

    result = ocr.predict(image_path)

    texts = []
    scores = []

    for page in result:
        data = page.json

        if isinstance(data, str):
            import json
            data = json.loads(data)

        ocr_data = data.get("res", data)

        page_texts = ocr_data.get("rec_texts", [])
        page_scores = ocr_data.get("rec_scores", [])

        for text in page_texts:
            if text:
                texts.append(text)

        for score in page_scores:
            scores.append(float(score))

    text = "\n".join(texts)

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return {
        "text": text,
        "confidence": round(confidence, 4),
    }
