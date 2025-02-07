# app/routes/chat_router.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.conversation import process_user_message
import logging
from app.models.schemas import ChatRequest, ChatResponse, JobPosting

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="사용자 메시지가 비어있습니다.")

    # 사용자 메시지를 처리
    try:
        response_text, user_profile = process_user_message(req)
        return ChatResponse(
            message=response_text,
            user_profile=user_profile,
            jobPostings=[],  # 필요하다면 일자리 정보도 포함
            type="info"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
