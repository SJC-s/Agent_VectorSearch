# app/models/schemas.py

from typing import Optional, Dict, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    클라이언트 → 서버로 들어오는 채팅 요청 정보
    """
    user_message: str
    session_id: str
    user_profile: Dict = {}

class JobPosting(BaseModel):
    """
    일자리 정보 예시 모델
    """
    id: str
    location: str
    company: str
    title: str
    salary: str
    workingHours: str
    description: str
    rank: int

class ChatResponse(BaseModel):
    """
    서버 → 클라이언트로 나가는 채팅 응답 정보
    """
    message: str
    type: str
    user_profile: Dict
    job_postings: List[JobPosting] = []

