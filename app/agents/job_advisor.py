# app/agents/job_advisor.py

import os
import re
import json
import logging
from typing import Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from app.models.profile import UserProfile
from app.services.vector_store_search import VectorStoreSearch
from app.core.prompts import SYSTEM_PROMPT
from app.models.schemas import StateDict

logger = logging.getLogger(__name__)

# 전역 프로필 (실서비스라면 세션별/DB 별도로 관리)
GLOBAL_USER_PROFILE = UserProfile()

# --- 기존 get_user_ner_runnable 정의 (함수 형태) ---
def get_user_ner_runnable() -> Runnable:
    """
    사용자 입력 예: "서울 요양보호사"
    -> LLM이 아래와 같이 JSON으로 추출:
       {"직무": "요양보호사", "지역": "서울", "연령대": ""}
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0.0
    )

    prompt = PromptTemplate(
        input_variables=["user_query"],
        template=(
            "사용자 입력: {user_query}\n\n"
            "아래 항목을 JSON으로 추출 (값이 없으면 빈 문자열로):\n"
            "- 직무\n"
            "- 지역\n"
            "- 연령대\n\n"
            "예:\n"
            "```json\n"
            "{{\"직무\": \"요양보호사\", \"지역\": \"서울\", \"연령대\": \"\"}}\n"
            "```\n"
        )
    )
    # PromptTemplate와 LLM을 체이닝하여 Runnable 반환
    return prompt | llm

###############################################################################
# (A) 노드 함수들
###############################################################################

def user_ner_node(state: Dict) -> Dict:
    """
    1) state["user_input"]에서 나이, 직무, 지역 등을 추출합니다.
    2) 추출한 NER 결과를 state["ner_result"]에 저장합니다.
    """
    user_message = state.get("user_input", "")
    logger.info(f"[user_ner_node] 입력: {user_message}")

    # state에 저장된 user_profile (예: {"age": "", "location": "", "jobType": ""})
    user_profile = state.get("user_profile", {})

    # get_user_ner_runnable()를 직접 호출 (self 없이)
    ner_chain = get_user_ner_runnable()
    ner_res = ner_chain.invoke({"user_query": user_message})
    # LLM 응답이 AIMessage라면 .content, 아니면 문자열 변환
    ner_str = getattr(ner_res, "content", str(ner_res))
    cleaned = ner_str.replace("```json", "").replace("```", "").strip()

    try:
        user_ner = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"[JobAdvisor] NER parse fail: {cleaned}")
        user_ner = {}

    logger.info(f"[JobAdvisor] 1) user_ner={user_ner}")

    # 프로필 보완: state에 저장된 user_profile의 값으로 채웁니다.
    if not user_ner.get("직무") and user_profile.get("jobType"):
        user_ner["직무"] = user_profile["jobType"]
    if not user_ner.get("지역") and user_profile.get("location"):
        user_ner["지역"] = user_profile["location"]
    if not user_ner.get("연령대") and user_profile.get("age"):
        user_ner["연령대"] = user_profile["age"]

    logger.info(f"[JobAdvisor] 1-1) 보완된 user_ner={user_ner}")

    # 최종 NER 결과를 state에 저장합니다.
    state["ner_result"] = user_ner
    return state

def profile_update_node(state: Dict) -> Dict:
    """
    1) state["ner_result"]를 GLOBAL_USER_PROFILE에 적용
    2) 프로필 요약문 state["profile_summary"]에 저장
    """
    ner_result = state.get("ner_result", {})
    if "age" in ner_result:
        GLOBAL_USER_PROFILE.update("age", ner_result["age"])
    if "location" in ner_result:
        GLOBAL_USER_PROFILE.update("location", ner_result["location"])
    if "job" in ner_result:
        jobs = GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])
        if ner_result["job"] not in jobs:
            jobs.append(ner_result["job"])
        GLOBAL_USER_PROFILE.update("preferred_jobs", jobs)

    summary = GLOBAL_USER_PROFILE.get_profile_summary()
    state["profile_summary"] = summary
    return state

def vector_search_node(state: Dict) -> Dict:
    """
    1) GLOBAL_USER_PROFILE에서 (직무, 지역) 가져옴
    2) vector_search 인스턴스를 사용해 채용 검색
    3) 결과(목록)를 state["job_postings"]에 저장
    """
    vector_search: VectorStoreSearch = state["vector_search"]  # main에서 주입
    # 프로필
    jobs = GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])
    loc = GLOBAL_USER_PROFILE.data.get("location", "")

    user_ner = {"직무": ",".join(jobs), "지역": loc}
    docs = vector_search.search_jobs(user_ner, top_k=5)

    job_list = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        job_list.append({
            "id": md.get("채용공고ID", f"doc_{i}"),
            "location": md.get("근무지역", ""),
            "company": md.get("회사명", ""),
            "title": md.get("채용제목", ""),
            "salary": md.get("급여조건", ""),
            "rank": i
        })
    state["job_postings"] = job_list
    return state

def final_response_node(state: Dict) -> Dict:
    """
    (선택) LLM을 사용하여 최종 응답 메시지를 생성
    1) user_input, profile_summary, job_postings를 종합해 답변
    2) state["final_answer"]에 저장
    """
    llm: ChatOpenAI = state["llm"]  # main에서 주입
    user_msg = state.get("user_input", "")
    summary = state.get("profile_summary", "")
    postings = state.get("job_postings", [])

    # LLM prompt
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    user_prompt = (
        f"사용자 발화: {user_msg}\n"
        f"프로필 요약: {summary}\n"
        f"검색된 채용공고 수: {len(postings)}\n"
        "이에 대한 최종 안내 문장을 만들어주세요."
    )
    resp = llm.invoke([system_msg, HumanMessage(content=user_prompt)])
    state["final_answer"] = resp.content
    return state

###############################################################################
# (B) 그래프 구성
###############################################################################
def build_job_advisor_graph(llm: ChatOpenAI, vector_search: VectorStoreSearch):
    """
    LangGraph로 노드를 연결:
    START -> user_ner_node -> profile_update_node 
          -> vector_search_node -> final_response_node -> END
    """
    builder = StateGraph(StateDict)

    # 노드 추가
    builder.add_node("user_ner", user_ner_node)
    builder.add_node("profile_update", profile_update_node)
    builder.add_node("vector_search", vector_search_node)
    builder.add_node("final_response", final_response_node)

    # 엣지 연결
    builder.add_edge(START, "user_ner")
    builder.add_edge("user_ner", "profile_update")
    builder.add_edge("profile_update", "vector_search")
    builder.add_edge("vector_search", "final_response")
    builder.add_edge("final_response", END)

    # Graph compile
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    # 그래프 실행 시 필요한 llm, vector_search를 state에 넣어주도록 함
    # => chat_router에서 run() 호출 시에 state={"llm": llm, "vector_search": vector_search, ...}

    return graph


def handle_chat(query: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    chat_router -> handle_chat
    1) NER 추출
    2) 프로필 업데이트
    3) 채용 검색
    4) 최종 응답
    5) 결과 딕셔너리 반환
    """
    logger.info(f"[JobAdvisor] handle_chat 호출. query={query}, user_profile={user_profile}")

    # (1) NER 추출
    ner_result = extract_user_ner(query)
    logger.info(f"[JobAdvisor] NER={ner_result}")

    # (2) 전역 프로필 업데이트
    update_profile(ner_result)
    logger.info(f"[JobAdvisor] 전역 프로필={GLOBAL_USER_PROFILE.data}")

    # (3) 채용 검색
    job_result = search_jobs()
    logger.info(f"[JobAdvisor] job_result={job_result}")

    # (4) 최종 답변
    answer = final_answer(query, job_result)

    # (5) 반환
    # 필요 시 job_result["jobPostings"] 리스트를 그대로 전달
    return {
        "message": answer if job_result["jobPostings"] else job_result["message"],
        "jobPostings": job_result["jobPostings"],
        "type": "jobPosting" if job_result["jobPostings"] else "info",
        "user_profile": user_profile
    }