from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "PaddleOCR FastAPI Service"
    database_url: str = (
        "postgresql+psycopg://ocr_user:ocr_password@db:5432/ocr_db"
    )

    upload_dir: str = "uploads"
    max_upload_size_mb: int = 20

    model_name: str = "PP-OCRv5_server"
    lang: str = "en"
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()