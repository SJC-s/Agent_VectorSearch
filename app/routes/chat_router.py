# app/routes/chat_router.py
import logging
from fastapi import APIRouter, Request
from app.models.schemas import ChatRequest, ChatResponse, JobPosting

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat/", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    /api/v1/chat/ 엔드포인트
    1) job_advisor_agent를 얻어온다.
    2) chat_request.user_message, user_profile을 전달해 에이전트 호출
    3) 결과(딕셔너리)를 ChatResponse로 변환
    """
    try:
        logger.info(f"[ChatRouter] 사용자의 메시지: {chat_request.user_message}")
        job_advisor_agent = request.app.state.job_advisor_agent
        if job_advisor_agent is None:
            logger.error("[ChatRouter] job_advisor_agent가 초기화되지 않음")
            return ChatResponse(
                message="서버가 준비되지 않았습니다.",
                jobPostings=[],
                type="error",
                user_profile={}
            )
        
        # 에이전트 호출(비동기)
        response = await job_advisor_agent.chat(
            query=chat_request.user_message,
            user_profile=chat_request.user_profile
        )
        logger.info("[ChatRouter] 에이전트 응답 완료")

        # jobPostings
        job_postings_list = []
        for idx, jp in enumerate(response.get("jobPostings", [])):
            job_postings_list.append(JobPosting(
                id=jp.get("id", "no_id"),
                location=jp.get("location", ""),
                company=jp.get("company", ""),
                title=jp.get("title", ""),
                salary=jp.get("salary", ""),
                workingHours=jp.get("workingHours", "정보없음"),
                description=jp.get("description", ""),
                rank=jp.get("rank", idx+1)
            ))

        # ChatResponse 모델 생성
        return ChatResponse(
            message=response.get("message", ""),
            jobPostings=job_postings_list,
            type=response.get("type", "info"),
            user_profile=response.get("user_profile", {})
        )

    except Exception as e:
        logger.error(f"[ChatRouter] 전체 처리 중 에러: {str(e)}", exc_info=True)
        return ChatResponse(
            message="처리 중 오류가 발생했습니다.",
            jobPostings=[],
            type="error",
            user_profile={}
        )
