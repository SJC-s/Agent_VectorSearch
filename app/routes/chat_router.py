# app/routes/chat_router.py
import logging
from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, JobPosting
from app.agents.job_advisor import handle_chat

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat/", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    /api/v1/chat/ 엔드포인트:
    1) Request 객체를 통해 app.state에서 job_advisor_graph와 vector_search_obj를 가져옵니다.
    2) chat_request를 통해 사용자 메시지 및 프로필 정보를 state에 담고, vector_search_obj도 함께 전달합니다.
    3) 그래프 실행 후, state["final_answer"] 및 state["job_postings"]를 ChatResponse로 반환합니다.
    """
    if not chat_request.user_message:
        raise HTTPException(status_code=400, detail="사용자 메시지가 필요합니다.")

    try:
        # (1) job_advisor 호출
        response_data = handle_chat(query=chat_request.user_message, user_profile=chat_request.user_profile)
        logger.info(f"[chat_endpoint] job_advisor result={response_data}")

        # (2) jobPostings 변환
        job_postings_raw = response_data.get("jobPostings", [])
        job_postings_list = []
        for idx, jp in enumerate(job_postings_raw, start=1):
            job_postings_list.append(JobPosting(
                id=jp.get("id", "no_id"),
                location=jp.get("location", ""),
                company=jp.get("company", ""),
                title=jp.get("title", ""),
                salary=jp.get("salary", ""),
                workingHours=jp.get("workingHours", "정보없음"),
                description="",
                rank=idx
            ))

        # (3) 최종 ChatResponse
        return ChatResponse(
            message=response_data.get("message", ""),
            jobPostings=job_postings_list,
            type=response_data.get("type", "info"),
            user_profile=response_data.get("user_profile", {})
        )
    except Exception as e:
        logger.error(f"[chat_endpoint] 처리 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
