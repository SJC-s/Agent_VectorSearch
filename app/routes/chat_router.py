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
        # app.state에서 vector_search 객체를 가져옵니다.
        vector_search_obj = getattr(request.app.state, "vector_search", None)
        if vector_search_obj is None:
            raise HTTPException(status_code=500, detail="vector_search 객체가 준비되지 않았습니다.")

        result = handle_chat(
            query=chat_request.user_message,
            user_profile=chat_request.user_profile,
            vector_search=vector_search_obj
        )
        logger.info(f"[chat_endpoint] handle_chat 결과: {result}")
        # result는 {"messages": ..., "job_postings": [...], "type": ..., "user_profile": ...} 형태

        # jobPostings 배열을 JobPosting 모델로 변환 (필요 시)
        job_postings_list = []
        for idx, jp in enumerate(result.get("job_postings", []), start=1):
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
        return ChatResponse(
            message=result.get("message", ""),
            job_postings=job_postings_list,
            type=result.get("type", "info"),
            user_profile=result.get("user_profile", {})
        )
    except Exception as e:
        logger.error(f"[chat_endpoint] 처리 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))