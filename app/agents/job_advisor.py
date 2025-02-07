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

# 툴 정의
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
