# app/agents/job_advisor.py
import os
import re
import json
import logging
from typing import Dict, Any, List, Annotated
from datetime import datetime

# LangChain & Tools
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool  # @tool 사용
from langchain.schema.runnable import Runnable

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add

# LangChain Prompts
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition

# 프로젝트 내 파일들
from app.models.profile import UserProfile
from app.models.schemas import JobPosting
from app.services.vector_store_search import VectorStoreSearch
from app.services.vector_store_ingest import VectorStoreIngest
from app.core.prompts import SYSTEM_PROMPT
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 전역 프로필 (실서비스라면 세션별/DB 별도로 관리)
GLOBAL_USER_PROFILE = UserProfile()


###############################################################################
# (A) StateDict Pydantic 모델 (추가 필드 허용)
###############################################################################
class StateDict(BaseModel):
    messages: Annotated[List[Any], add]  # 다중 값 업데이트 허용
    user_profile: Dict[str, Any]
    user_ner: Dict[str, Any] = {}         # NER 결과
    profile_summary: str = ""             # 프로필 요약
    job_postings: List[Any] = []          # 채용 공고 목록
    final_answer: str = ""                # 최종 응답

    class Config:
        extra = "allow"  # 추가 필드 허용

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)
    
    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

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

    llm = setup_openai(0)

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

def setup_openai(temperature: float):
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        streaming=True
    )
    return llm

###############################################################################
# (A) Tool 함수들
###############################################################################

@tool
def user_ner_tool(input_data: dict) -> Dict[str, Any]:
    """
    (1) 사용자 입력 NER 추출
    (1-1) NER 데이터가 없거나 누락된 항목은 user_profile (age, location, jobType)로 보완
    입력 데이터: {"user_message": str, "user_profile": dict}
    """
    user_message = input_data.get("messages", "")
    user_profile = input_data.get("user_profile", {})

    logger.info(f"[user_ner_tool] user_message={user_message}")

    # 입력 검증 강화
    if not user_message:  # 빈 입력 처리
        logger.warning("Empty user message received")
        return {"직무": "", "지역": "", "연령대": ""}

    # LLM 기반 NER 호출 (여기서는 get_user_ner_runnable을 사용)
    ner_chain = get_user_ner_runnable()
    ner_res = ner_chain.invoke({"user_query": user_message})
    # 결과 처리 개선
    ner_str = getattr(ner_res, "content", str(ner_res))
    cleaned =  ner_str.replace("```json", "").replace("```", "").strip()

    logger.info(f"[user_ner_tool] 중간 NER 결과: {cleaned}")


    try:
        user_ner = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"[user_ner_tool] NER parse fail: {cleaned}")
        user_ner = {}

    # 프로필 보완
    if not user_ner.get("직무") and user_profile.get("jobType"):
        user_ner["직무"] = user_profile["jobType"]
    if not user_ner.get("지역") and user_profile.get("location"):
        user_ner["지역"] = user_profile["location"]
    if not user_ner.get("연령대") and user_profile.get("age"):
        user_ner["연령대"] = user_profile["age"]

    logger.info(f"[user_ner_tool] 최종 NER 결과: {user_ner}")
    return user_ner

@tool
def profile_update_tool(ner_data: Dict[str, str]) -> str:
    """
    NER 데이터(예: {"직무":"경비","지역":"서울","연령대":"60대"})를 받아
    GLOBAL_USER_PROFILE를 업데이트하고, 요약 문자열을 반환
    """
    if "연령대" in ner_data and ner_data["연령대"]:
        # "60대" -> 60
        number_match = re.search(r"(\d+)", ner_data["연령대"])
        if number_match:
            GLOBAL_USER_PROFILE.update("age", number_match.group(1))

    if "직무" in ner_data and ner_data["직무"]:
        jobs = GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])
        if ner_data["직무"] not in jobs:
            jobs.append(ner_data["직무"])
        GLOBAL_USER_PROFILE.update("preferred_jobs", jobs)

    if "지역" in ner_data and ner_data["지역"]:
        GLOBAL_USER_PROFILE.update("location", ner_data["지역"])
    
    logger.info(f"[profile_update_tool] 최종 GLOBAL_USER_PROFILE 결과: {GLOBAL_USER_PROFILE}")
    return GLOBAL_USER_PROFILE.get_profile_summary()

@tool
def vector_search_tool(_: str) -> Dict[str, Any]:
    """
    GLOBAL_USER_PROFILE 기반으로 벡터 검색
    """
    vs = VectorStoreSearch(vectorstore=None)  # 실제 Chroma 인스턴스
    jobs = GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])
    loc = GLOBAL_USER_PROFILE.data.get("location", "")

    user_ner = {"직무": ",".join(jobs), "지역": loc}
    docs = vs.search_jobs(user_ner, top_k=5)

    if not docs:
        return {
            "job_postings": [],
            "messages": "조건에 맞는 채용 공고가 없습니다."
        }

    job_postings = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        job_postings.append({
            "id": md.get("채용공고ID", f"doc_{i}"),
            "location": md.get("근무지역", ""),
            "company": md.get("회사명", ""),
            "title": md.get("채용제목", ""),
            "salary": md.get("급여조건", ""),
            "rank": i
        })

    logger.info(f"[vector_search_tool] 최종 job_postings 결과: {job_postings}")

    return {
        "job_postings": job_postings,
        "messages": "채용 공고 검색 결과"
    }

@tool
def final_response_tool(input_data: Dict) -> str:
    """
    LLM을 사용하여 최종 답변 문자열을 생성
    input_data: {"messages": "...", "user_profile": "...", "job_postings": [...]}
    """
    messages = input_data.get("messages", "")
    user_profile = input_data.get("user_profile", "")
    postings = input_data.get("job_postings", [])

    llm = setup_openai(0.5)
    
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    user_prompt = (
        f"사용자 발화: {messages}\n"
        f"프로필 요약: {user_profile}\n"
        f"검색된 채용공고 수: {len(postings)}\n"
        "위 정보를 종합해 한 문장으로 안내해줘."
    )
    resp = llm.invoke([system_msg, HumanMessage(content=user_prompt)])

    logger.info(f"[final_response_tool] 최종 resp 결과: {resp}")

    return resp.content

###############################################################################
# (B) 각 Node 함수: Tool을 호출
###############################################################################
def user_ner_node(state: Dict) -> Dict:
    """
    user_ner_tool 호출 -> state["user_ner"]에 저장
    """
    user_input = state["messages"]
    user_profile = state["user_profile"]

    logger.info(f"[user_ner_node] user_input={user_input}")
    logger.info(f"[user_ner_node] user_profile={user_profile}")

    # 수정 후 ✅ input_data 필드 추가
    ner_dict = user_ner_tool.invoke({
        "input_data": {  # Pydantic 모델 구조에 맞게 감싸기
            "messages": user_input,
            "user_profile": user_profile
        }
    })
    # 수정된 부분: 새 객체 생성 후 필드 추가
    state.user_ner = ner_dict
    logger.info(f"[user_ner_node] state={state}")
    return state

def profile_update_node(state: Dict) -> Dict:
    """
    profile_update_tool 호출 -> state["profile_summary"]에 저장
    """
    ner_dict = state["user_ner"]
    summary = profile_update_tool.invoke({"ner_data": ner_dict})
    state.profile_summary = summary
    logger.info(f"[profile_update_node] 업데이트 후 state: {state}")
    return state

def vector_search_node(state: Dict) -> Dict:
    """
    state["vector_search"] (메인에서 주입한 객체)를 사용하여 채용 검색을 수행하고,
    결과를 state["job_postings"]에 저장합니다.
    """
    vector_search = VectorStoreSearch(vectorstore=VectorStoreIngest().setup_vector_store())  # ✅ 직접 초기화
    if not vector_search:
        raise ValueError("vector_search 객체가 state에 없습니다.")
    
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
    state.job_postings = job_list
    logger.info(f"[vector_search_node] 업데이트 후 state: {state}")
    return state

def final_response_node(state: Dict) -> Dict:
    """
    최종 응답 도출: 사용자 입력, 프로필 요약, 검색 결과를 종합하여 final_response_tool을 호출하고,
    결과를 state["final_answer"]에 저장합니다.
    """
    messages = state["messages"]
    user_profile = state["user_profile"]
    postings = state["job_postings"]

    logger.info(f"[final_response_node] postings={postings}")

    payload = {
        "input_data": {  # Pydantic 모델 구조에 맞게 감싸기
            "messages": messages,
            "user_profile": user_profile,
            "job_postings": postings
        }
    }
    answer = final_response_tool.invoke(payload)
    
    # jobPostings 배열을 JobPosting 모델로 변환 (필요 시)
    job_postings_list = []
    for idx, jp in enumerate(postings, start=1):
        job_postings_list.append(JobPosting(
            id=str(jp.get("id", "no_id")),  # id를 문자열로 변환
            location=jp.get("location", ""),
            company=jp.get("company", ""),
            title=jp.get("title", ""),
            salary=jp.get("salary", ""),
            workingHours=jp.get("workingHours", "정보없음"),
            description="",
            rank=idx
        ))
    state["job_postings"] = job_postings_list
    state["final_answer"] = answer
    logger.info(f"[final_response_node] 최종 state: {state}")
    return state

###############################################################################
# (C) 그래프 생성
###############################################################################
def build_job_advisor_graph(llm: ChatOpenAI, vector_search: VectorStoreSearch) -> StateGraph:
    """
    LangGraph로 노드를 연결:
    START -> user_ner_node -> profile_update_node
          -> vector_search_node -> final_response_node -> END
    """
    # 1) LLM 준비
    llm = setup_openai(0.5)
    tools = [vector_search_tool, profile_update_tool, final_response_tool]

    llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(StateDict)

    # 3) 메인 챗봇 노드
    def chatbot_node(state: Dict):
        try:
            # 시스템 메시지와 사용자 프로필 정보 추가
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            
            # 기존 메시지 추가
            if hasattr(state, 'messages') and state.messages:
                for msg in state.messages:
                    if isinstance(msg, (HumanMessage, SystemMessage, AIMessage)):
                        messages.append(msg)
                    elif isinstance(msg, dict) and 'content' in msg:
                        if msg.get('role') == 'user':
                            messages.append(HumanMessage(content=msg['content']))
                        elif msg.get('role') == 'assistant':
                            messages.append(AIMessage(content=msg['content']))
                        elif msg.get('role') == 'system':
                            messages.append(SystemMessage(content=msg['content']))
            
            # 프로필 정보를 문자열로 변환하여 컨텍스트에 추가
            if hasattr(state, 'user_profile') and state.user_profile:
                profile_info = f"\n현재 사용자 정보:\n{str(state.user_profile)}"
                messages.append(SystemMessage(content=profile_info))
            
            # LLM 호출
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
            
        except Exception as e:
            print(f"챗봇 노드 처리 중 오류: {str(e)}")
            return {"messages": [AIMessage(content="죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 다시 한 번 말씀해 주시겠어요?")]}

    # 노드 등록
    builder.add_node("chatBot", chatbot_node)
    builder.add_node("userNer", user_ner_node)
    builder.add_node("profileUpdate", profile_update_node)
    builder.add_node("vectorSearch", vector_search_node)
    builder.add_node("finalResponse", final_response_node)

    # 도구 노드
    tool_node = ToolNode(tools=tools)
    builder.add_node("tools", tool_node)

    # 엣지 연결
    # 조건부 라우팅
    builder.add_edge("tools", "chatBot")
    builder.add_edge(START, "userNer")
    builder.add_edge("userNer", "chatBot")
    builder.add_conditional_edges("chatBot", tools_condition)
    builder.add_edge("userNer", "profileUpdate")
    builder.add_edge("profileUpdate", "vectorSearch")
    builder.add_edge("vectorSearch", "finalResponse")
    builder.add_edge("finalResponse", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph


###############################################################################
# (D) 통합 처리 함수: handle_chat
###############################################################################
def handle_chat(query: str, user_profile: Dict[str, Any], vector_search: VectorStoreSearch) -> Dict[str, Any]:
    """
    chat_router에서 호출하는 함수.
    1) 사용자 메시지와 프로필 정보를 받아, 상태(state)를 구성
    2) build_job_advisor_graph()를 호출하여 그래프를 생성
    3) 그래프 실행을 통해 모든 노드를 순차적으로 호출하고, 최종 응답을 state에 저장
    4) 최종 결과 딕셔너리를 반환
    """
    logger.info(f"[handle_chat] query={query}, user_profile={user_profile}")
    # StateDict를 사용하여 상태를 초기화 (추가 필드 vector_search 포함)
    state = StateDict(
        messages=[HumanMessage(content=query)],
        user_profile=user_profile
    )
    logger.info(f"[handle_chat] state 시작: {state}")

    llm_instance = setup_openai(0.5)

    # 그래프 빌드
    graph = build_job_advisor_graph(llm=llm_instance, vector_search=vector_search)

    logger.info(f"[handle_chat] state 중간: {state}")


    # 동기 실행 (stream을 사용하여 상태를 업데이트)
    result  = graph.invoke(
        state,
        {"configurable": {"thread_id": "demo-user"}},
        # stream_mode="values"
    )
    logger.info(f"[handle_chat] result 결과: {result}")
    # 결과 업데이트
    state.user_ner = result.get("user_ner", {})
    state.profile_summary = result.get("profile_summary", "")
    state.job_postings = result.get("job_postings", [])
    state.final_answer = result.get("final_answer", "")


    final_answer = state["final_answer"]
    job_postings = state["job_postings"]
    msg_type = "jobPosting" if job_postings else "info"

    return {
        "message": final_answer,
        "jobPostings": job_postings,
        "type": msg_type,
        "user_profile": user_profile
    }