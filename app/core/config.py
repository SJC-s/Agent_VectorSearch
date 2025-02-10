# app/core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_BASE_URL: str = "http://localhost:8000/api/v1"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",  # React 기본 개발 서버
        "http://localhost:5173",  # Vite 개발 서버
        "http://127.0.0.1:5173"   # Vite 개발 서버 (IP 주소)
    ]

settings = Settings() 

# from sqlalchemy import create_engine, text
# engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_size=5, max_overflow=10)