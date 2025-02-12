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
        allow_mutation = True  # 모델 변경 가능하게 설정

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

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
def user_ner_tool(state: dict) -> Dict[str, Any]:
    """
    (1) 사용자 입력 NER 추출
    (1-1) NER 데이터가 없거나 누락된 항목은 user_profile (age, location, jobType)로 보완
    입력 데이터: {"user_message": str, "user_profile": dict}
    """
    try:
        # 메시지 추출
        state_dict = StateDict(**state)
        user_message = state_dict.messages
        user_profile = state_dict.user_profile

        logger.info(f"[user_ner_tool] user_message={user_message}")

        # 입력 검증 강화
        if not user_message:  # 빈 입력 처리
            logger.warning("Empty user message received")
            return {"직무": "", "지역": "", "연령대": ""}

        # LLM 기반 NER 체인인
        ner_chain = get_user_ner_runnable()
        ner_res = ner_chain.invoke({"user_query": user_message})
        ner_str = getattr(ner_res, "content", str(ner_res))
        cleaned =  ner_str.replace("```json", "").replace("```", "").strip()

        logger.info(f"[user_ner_tool] 중간 NER 결과: {cleaned}")

        try:
            user_ner_result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"[user_ner_tool] NER parse fail: {cleaned}")
            user_ner = {}

        # 프로필 보완
        user_ner_result = {
            "직무": user_ner.get("직무") or user_profile.get("jobType", ""),
            "지역": user_ner.get("지역") or user_profile.get("location", ""),
            "연령대": user_ner.get("연령대") or user_profile.get("age", "")
        }

        logger.info(f"[user_ner_tool] 최종 NER 결과: {user_ner_result}")
        return {"user_ner": user_ner_result}
    except Exception as e:
        logger.error(f"NER 툴 오류: {str(e)}")
        return {"user_ner": {}}

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
def vector_search_tool(state: StateDict) -> Dict[str, Any]:
    """벡터 검색 툴 개선"""
    try:
        vs = VectorStoreSearch(vectorstore=None)  # 실제 Chroma 인스턴스
        user_ner = state.get("user_ner", {})
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
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return {"job_postings": []}

@tool
def final_response_tool(state: StateDict) -> Dict[str, Any]:
    """
    LLM을 사용하여 최종 답변 문자열을 생성
    input_data: {"messages": "...", "user_profile": "...", "job_postings": [...]}
    """
    try:
        messages = state.get("messages", [])
        user_profile = state.get("user_profile", {})
        postings = state.get("job_postings", [])

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

        return resp
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return {
            "final_answer": "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다.",
            "messages": [AIMessage(content="응답 생성 중 오류가 발생했습니다.")]
        }
    

###############################################################################
# (B) 각 Node 함수: Tool을 호출
###############################################################################
# Simplified node functions that just call tools
def user_ner_node(state: StateDict) -> Dict:
    """Call user_ner_tool and update state"""
    try:
        result = user_ner_tool.invoke(state)
        return StateDict(**{**state, **result})  # 결과 병합
    except Exception as e:
        logger.error(f"user_ner_node 오류: {str(e)}")
        return state

def profile_update_node(state: Dict) -> Dict:
    """Call profile_update_tool and update state"""
    result = profile_update_tool.invoke(state)
    return {**state, **result}

def vector_search_node(state: Dict) -> Dict:
    """Call vector_search_tool and update state"""
    result = vector_search_tool.invoke(state)
    return {**state, **result}

def final_response_node(state: Dict) -> Dict:
    """Call generate_response_tool and update state"""
    result = final_response_tool.invoke(state)
    return {**state, **result}

# 커스텀 조건 함수 구현
def custom_tools_condition(state: StateDict) -> str:
    """툴 호출 여부 판단 로직 강화"""
    last_message = state["messages"][-1]
    return "tools" if hasattr(last_message, 'tool_calls') else END

def chatbot_node(state: StateDict, llm_with_tools: ChatOpenAI) -> dict:
    """
    챗봇 노드는 시스템 메시지와 사용자 프로필 정보를 추가하여 LLM을 호출하고,
    그 결과를 state["messages"]에 저장합니다.
    이 노드는 그래프 외부에서 별도로 호출됩니다.
    """
    try:
        # 메시지 구성
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
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
        # 프로필 정보 접근 방식 변경
        if "user_profile" in state and state["user_profile"]:
            profile_info = f"\n현재 사용자 정보:\n{str(state['user_profile'])}"
            messages.append(SystemMessage(content=profile_info))
        
        # 도구 호출 가능한 LLM 실행
        response = llm_with_tools.invoke(messages)

        logger.info(f"[chat_node] user_ner 확인: {state.get('user_ner', {})}")
        
        # 툴 호출 감지
        if hasattr(response, "tool_calls") and response.tool_calls:
            return {
                "messages": [response],
                "tool_calls": [tc for tc in response.tool_calls]
            }
        
        # 일반 응답
        return {"messages": [response]}
    except Exception as e:
        print(f"챗봇 노드 처리 중 오류: {str(e)}")
        return {"messages": [AIMessage(content="죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 다시 한 번 말씀해 주시겠어요?")]}


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
    tools = [user_ner_tool, final_response_tool]

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
            
            logger.info(f"[chat_node] ner_result 확인: {state['ner_result']}")
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
    builder.add_edge(START, "chatBot")
    builder.add_conditional_edges("chatBot", tools_condition)
    builder.add_edge("chatBot", "userNer")
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
    state = {
        "messages": [HumanMessage(content=query)],
        "user_profile": user_profile,
        "user_input": query,  # 반드시 추가: user_ner_node에서 사용됨
        "vector_search": vector_search
    }
    llm_instance = setup_openai(0.5)

    # 그래프 빌드
    graph = build_job_advisor_graph(llm=llm_instance, vector_search=vector_search)

    logger.info(f"[handle_chat] graph 결과: {graph}")


    # 동기 실행 (stream을 사용하여 상태를 업데이트)
    events = graph.stream(
        state,
        {"configurable": {"thread_id": "demo-user"}},
        stream_mode="values"
    )
    for _ in events:
        pass

    logger.info(f"[handle_chat] state 결과: {state}")

    final_answer = state.get("final_answer", "")
    job_postings = state.get("job_postings", [])
    msg_type = "jobPosting" if job_postings else "info"

    return {
        "message": final_answer,
        "jobPostings": job_postings,
        "type": msg_type,
        "user_profile": user_profile
    }