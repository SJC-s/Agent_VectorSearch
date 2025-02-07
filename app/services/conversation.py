# app/services/conversation.py
from typing import Tuple, Dict
from app.models.schemas import ChatRequest
from app.agents.job_advisor import run_job_advisor
import logging

logger = logging.getLogger(__name__)

def process_user_message(req: ChatRequest) -> Tuple[str, Dict]:
    """
    1) 사용자 메시지를 받아서 프로필 업데이트 등
    2) LangGraph 에이전트(job_advisor.py)에 전달
    3) 에이전트 응답을 받아 반환
    """
    session_id = req.session_id
    user_message = req.user_message
    user_profile = req.user_profile  # dict
    
    # 필요 시 user_profile 업데이트: (ex: user_profile["age"] = 63 ...)
    # 여기서는 생략
    
    response_text = run_job_advisor(user_message, user_profile)
    
    # response_text: 최종 AI의 답변
    # user_profile: 업데이트된 프로필(필요시 반환)
    return response_text, user_profile
