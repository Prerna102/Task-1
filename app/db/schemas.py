from typing import Optional

from pydantic import BaseModel



class OCRUploadResponse(BaseModel):
    """Response returned after uploading one file."""

    success: bool
    job_id: str
    filename: str
    status: str
    message: str

class OCRStatusResponse(BaseModel):
    """Response returned when checking one OCR job."""

    success: bool
    job_id: str
    filename: str
    status: str
    text: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None





class MultipleOCRUploadResponse(BaseModel):
    success: bool
    files: list[OCRUploadResponse]
    message: str
