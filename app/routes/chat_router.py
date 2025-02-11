# app/routes/chat_router.py
import logging
from fastapi import APIRouter, Request, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, JobPosting, StateDict
from app.agents.job_advisor import ChatHandler
from typing import List, Dict
from pydantic import BaseModel
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from app.agents.job_advisor import build_graph


logger = logging.getLogger(__name__)

router = APIRouter()



@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    /api/v1/chat/ 엔드포인트:
    1) Request 객체를 통해 app.state에서 job_advisor_graph와 vector_search_obj를 가져옵니다.
    2) chat_request를 통해 사용자 메시지 및 프로필 정보를 state에 담고, vector_search_obj도 함께 전달합니다.
    3) 그래프 실행 후, state["final_answer"] 및 state["job_postings"]를 ChatResponse로 반환합니다.
    """
    if not request.user_message:
        raise HTTPException(status_code=400, detail="사용자 메시지가 필요합니다.")
    
    try:
        # 사용자 프로필 초기화 또는 업데이트
        user_profile = request.user_profile
        
        # 메시지에서 사용자 정보 추출 및 프로필 업데이트
        info = extract_user_info_from_text(request.user_message)
        for key, value in info.items():
            if key in user_profile:
                if isinstance(user_profile[key], list):
                    if isinstance(value, list):
                        user_profile[key].extend(value)
                    else:
                        user_profile[key].append(value)
                else:
                    user_profile[key] = value

        # jobType을 preferred_jobs에 추가
        if user_profile.get("jobType"):
            if "preferred_jobs" not in user_profile:
                user_profile["preferred_jobs"] = []
            if user_profile["jobType"] not in user_profile["preferred_jobs"]:
                user_profile["preferred_jobs"].append(user_profile["jobType"])

        # 상태 업데이트
        state = StateDict(
            messages=[HumanMessage(content=request.user_message)],
            user_profile=user_profile
        )
        
        graph = build_graph()

        # 그래프 실행
        response_data = graph.stream(
            state.model_dump(),  # dict() 대신 model_dump() 사용
            {"configurable": {"thread_id": "demo-user"}},
            stream_mode="values"
        )
        
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
        print(f"chat_endpoint 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        )
