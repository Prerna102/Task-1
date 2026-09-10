from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, Text, String

from app.database import Base


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(
        String(255),
        nullable=False,
    )

    stored_filename = Column(
        String(255),
        nullable=False,
    )

    extracted_text = Column(
        Text,
        nullable=False,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
