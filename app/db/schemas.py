from datetime import datetime

from pydantic import BaseModel


class OCRResponse(BaseModel):
    id: int
    filename: str
    text: str
    confidence: float | None
    created_at: datetime | None

    class Config:
        from_attributes = True
