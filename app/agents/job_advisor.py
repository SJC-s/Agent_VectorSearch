# app/agents/job_advisor.py
from langchain_core.tools import tool
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Dict
import logging

from app.core.prompts import SYSTEM_PROMPT
from app.services.vector_store_ingest import VectorStoreIngest
from app.services.vector_store_search import VectorStoreSearch

logger = logging.getLogger(__name__)



###############################################################################
# (A) NER 추출용 함수
###############################################################################
def get_user_ner_runnable(self) -> Runnable:
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
            "json\n"
            "{{\"직무\": \"요양보호사\", \"지역\": \"서울\", \"연령대\": \"\"}}\n"
            "\n"
        )
    )
    return prompt | llm

def _extract_user_ner(self, user_message: str, user_profile: Dict[str, str]) -> Dict[str, str]:
    """
    (1) 사용자 입력 NER 추출
    (1-1) NER 데이터가 없거나 누락된 항목은 user_profile (age, location, jobType)로 보완
    """
    # 1) 사용자 입력 NER
    ner_chain = self.get_user_ner_runnable()
    ner_res = ner_chain.invoke({"user_query": user_message})
    ner_str = ner_res.content if hasattr(ner_res, "content") else str(ner_res)
    cleaned = ner_str.replace("```json", "").replace("```", "").strip()

    try:
        user_ner = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"[JobAdvisor] NER parse fail: {cleaned}")
        user_ner = {}

    logger.info(f"[JobAdvisor] 1) user_ner={user_ner}")

    # 1-1) 프로필 보완
    # user_profile: {"age":"", "location":"", "jobType":""}
    if not user_ner.get("직무") and user_profile.get("jobType"):
        user_ner["직무"] = user_profile["jobType"]
    if not user_ner.get("지역") and user_profile.get("location"):
        user_ner["지역"] = user_profile["location"]
    if not user_ner.get("연령대") and user_profile.get("age"):
        user_ner["연령대"] = user_profile["age"]

    logger.info(f"[JobAdvisor] 1-1) 보완된 user_ner={user_ner}")
    return user_ner
# 툴 정의
def extract_user_info_from_text(text: str) -> dict:
    """대화 내용에서 사용자 정보를 추출"""
    info = {}
    
    # 나이 추출
    age_pattern = re.search(r"(\d+)[세살]", text)
    if age_pattern:
        info["age"] = int(age_pattern.group(1))
    
    # 직종 키워드 추출
    job_keywords = ["경비", "운전", "사무", "강사"]
    for job in job_keywords:
        if job in text:
            if "preferred_jobs" not in info:
                info["preferred_jobs"] = []
            info["preferred_jobs"].append(job)
    
    return info

@tool
def update_user_profile(text: str) -> str:
    """사용자 프로필 정보를 추출하고 업데이트하는 도구"""
    info = extract_user_info_from_text(text)
    return str(info)



@tool
def vector_search_tool(query: str) -> str:
    """벡터 DB 유사도 검색"""
    docs = VectorStoreSearch.search_jobs(query, k=3)
    if not docs:
        return "검색 결과가 없습니다."
    return "\n".join(
        [f"[{i+1}] {doc.page_content[:80]}..." for i, doc in enumerate(docs)]
    )

# 실제 LangGraph 빌드
def build_graph():
    # LangGraph를 이용한 간단한 워크플로우 구성
    def chatbot_node(state):
        messages = state["messages"]
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        # 여기서는 단순히 마지막 사용자 메시지를 회신하는 임시 응답 생성
        last_user_msg = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
        return {"messages": [AIMessage(content=f"에이전트 응답(임시): {last_user_msg}")]}
    
    graph = StateGraph(dict)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge("start", "chatbot")
    return graph.compile()

agent_graph = build_graph()

def run_job_advisor(user_message: str, user_profile: Dict) -> str:
    # LangGraph 에이전트 실행: 사용자 메시지와 프로필을 컨텍스트로 전달
    messages = [HumanMessage(content=user_message)]
    state = {
        "messages": messages,
        "user_profile": user_profile
    }
    events = agent_graph.run(state)
    final_resp = None
    for event in events:
        if "messages" in event and event["messages"]:
            for m in event["messages"]:
                if isinstance(m, AIMessage):
                    final_resp = m.content
    return final_resp or "응답을 생성하지 못했습니다."
