import os
import re
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# LangChain & Tools
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema.runnable import Runnable

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain Components
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition

# Project Modules
from app.models.profile import UserProfile
from app.services.vector_store_search import VectorStoreSearch
from app.core.prompts import SYSTEM_PROMPT
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# 1. Enhanced State Model
class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)  # Dict 타입 명시
    user_profile: Dict = Field(default_factory=dict)
    ner_result: Dict = Field(default_factory=dict)
    job_postings: List[Dict] = Field(default_factory=list)
    
    model_config = ConfigDict(
        extra="ignore",
        frozen=False,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def update(self, **kwargs):
        return self.model_copy(update=kwargs)
    
    @classmethod
    def from_langchain_messages(cls, messages: list):
        """LangChain 메시지 객체를 사전으로 변환"""
        return cls(
            messages=[msg.dict() for msg in messages],
            user_profile={},
            ner_result={},
            job_postings=[]
        )

    @field_validator('messages')
    def validate_messages(cls, v):
        """메시지 유효성 검사 강화"""
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("Messages must be dictionaries")
            if 'content' not in msg:
                raise ValueError("Message missing 'content' field")
        return v

# 2. Optimized Tools Implementation
class CoreTools:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    @tool
    def dynamic_ner_tool(self, input_data: Dict) -> Dict:
        """Enhanced NER extraction with fallback logic"""
        try:
            # 메시지 구조 수정
            messages = input_data.get("messages", "")
            profile = input_data.get("user_profile", {})
            
            ner_chain = self._get_ner_runnable()
            response = ner_chain.invoke({"user_query": messages})
            ner_data = self._parse_ner_response(response)
            
            return {
                "job": ner_data.get("직무") or profile.get("jobType", ""),
                "location": ner_data.get("지역") or profile.get("location", ""),
                "age": ner_data.get("연령대") or profile.get("age", "")
            }
        except Exception as e:
            logger.error(f"NER Error: {str(e)}")
            return {"job": "", "location": "", "age": ""}

    @tool
    def smart_search_tool(self, search_criteria: Dict) -> Dict:
        """Vector store search with dynamic initialization"""
        try:
            vs = VectorStoreSearch(vectorstore=self.vector_store)
            results = vs.search_jobs(search_criteria, top_k=5)
            return self._format_search_results(results)
        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
            return {"job_postings": [], "message": "검색 중 오류 발생"}

    # Helper methods
    def _get_ner_runnable(self) -> Runnable:
        """NER processing chain"""
        return PromptTemplate(
            template=(
                "Extract from: {user_query}\n"
                "Output JSON with keys: 직무, 지역, 연령대\n"
                "Example: {{\"직무\": \"요양보호사\", \"지역\": \"서울\"}}"
            ),
            input_variables=["user_query"]
        ) | ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _parse_ner_response(self, response: Any) -> Dict:
        """Parse LLM NER response"""
        try:
            content = getattr(response, "content", "")
            return json.loads(content.strip("`json\n"))
        except json.JSONDecodeError:
            return {}

    def _format_search_results(self, docs: List) -> List[Dict]:
        """Format vector search results"""
        return [
            {
                "id": doc.metadata.get("채용공고ID", f"doc_{idx}"),
                "title": doc.metadata.get("채용제목", ""),
                "company": doc.metadata.get("회사명", ""),
                "location": doc.metadata.get("근무지역", ""),
                "salary": doc.metadata.get("급여조건", "")
            } for idx, doc in enumerate(docs)
        ]


# 3. Intelligent Graph Builder (수정된 부분)
class JobAdvisorGraph:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.tools = CoreTools(vector_store)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
    def build(self):
        builder = StateGraph(AgentState)

        # 노드 구성
        builder.add_node("llm_router", self.llm_router)
        builder.add_node("tool_executor", ToolNode([
            self.tools.dynamic_ner_tool,
            self.tools.smart_search_tool
        ]))
        builder.add_node("response_generator", self.generate_response)

        # 엣지 구성
        builder.add_edge(START, "llm_router")
        builder.add_conditional_edges(
            "llm_router",
            tools_condition,
            {
                "tool_executor": "tool_executor",
                "final_response": "response_generator",
                "__end__": END  # LangGraph의 정식 END 노드 사용
            }
        )
        builder.add_edge("tool_executor", "llm_router")
        builder.add_edge("response_generator", END)

        return builder.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["tool_executor"]
        )

    def llm_router(self, state: AgentState) -> Dict:
        """Dynamic routing decision maker"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state.messages[-1]["content"])
        ]
        return self.llm.invoke(messages)

    def generate_response(self, state: AgentState) -> Dict:
        """응답 생성 로직 수정"""
        # 사전 → LangChain 메시지 변환
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "user_profile": state.user_profile,
                "job_postings": state.job_postings,
                "messages": state.messages
            }, ensure_ascii=False))
        ]
        
        response = self.llm.invoke(messages)
        return {"messages": [response.dict()]}  # dict() 사용

# 4. Unified Chat Handler (수정된 부분)
class ChatHandler:
    def __init__(self, vector_store):
        self.graph = JobAdvisorGraph(vector_store).build()
        self.state_model = AgentState

    def process_query(self, query: str, profile: Dict) -> Dict:
        # 초기 메시지 변환
        initial_message = HumanMessage(content=query)
                
        initial_state = self.state_model(
            messages=[initial_message],  # 사전 형식으로 저장
            user_profile=profile
        )

        events = self.graph.stream(  # 비동기 스트림 사용
            initial_state.model_dump(),
            {"configurable": {"thread_id": "user_session"}},
            stream_mode="values"
        )

        final_state = None
        for event in events:
            final_state = event

        return {
            "message": final_state["messages"][-1].get("content", ""),  # Dict 스타일 접근
            "job_postings": final_state["job_postings"],  # 키 기반 접근
            "user_profile": final_state["user_profile"]
        }
