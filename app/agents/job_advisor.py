# app/agents/job_advisor.py
import os
import re
import json
import logging
from typing import Dict, Any, List
from langchain_core.documents import Document
from datetime import datetime

# LangChain & OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate  # 올바른 경로로 수정
from langchain.schema.runnable import Runnable

# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.types import NodeFunction


# Vector 검색 서비스 (별도 파일에서 구현했다고 가정)
from app.models.profile import UserProfile
from app.services.vector_store_search import VectorStoreSearch
from app.core.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ========================================================================
# (A) 사용자 프로필 클래스
# ========================================================================
# 전역(또는 세션별)으로 사용자 프로필 객체 보관 예시(실서비스라면 세션별/DB별로 관리)
GLOBAL_USER_PROFILE = UserProfile()

# ========================================================================
# (B) NER 추출용: LangChain Runnable
# ========================================================================

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
            "{\"직무\": \"요양보호사\", \"지역\": \"서울\", \"연령대\": \"\"}\n"
            "```\n"
        )
    )
    # PromptTemplate와 LLM을 체이닝
    return prompt | llm

def _extract_user_ner(user_query: str, profile_data: Dict[str, Any]) -> Dict[str, str]:
    """
    1) get_user_ner_runnable() 로 LLM NER 추출
    2) 프로필 데이터와 merge
    """
    ner_chain = get_user_ner_runnable()
    ner_res = ner_chain.invoke({"user_query": user_query})

    # ner_res가 ChatOpenAI의 답변(AIMessage)이면 .content, 그렇지 않으면 str(ner_res)
    ner_str = getattr(ner_res, "content", str(ner_res))
    ner_str = ner_str.replace("```json", "").replace("```", "").strip()

    try:
        user_ner = json.loads(ner_str)
    except json.JSONDecodeError:
        logger.warning(f"NER parse fail. Raw = {ner_str}")
        user_ner = {}

    # 값이 없으면 프로필에서 보완
    if not user_ner.get("직무") and profile_data.get("preferred_jobs"):
        # preferred_jobs가 여러개면 첫 번째만 잡아둠
        user_ner["직무"] = profile_data["preferred_jobs"][0]

    if not user_ner.get("지역") and profile_data.get("location"):
        user_ner["지역"] = profile_data["location"]

    if not user_ner.get("연령대") and profile_data.get("age"):
        # age -> "60대"로 변환 예시
        age_val = profile_data["age"]
        user_ner["연령대"] = f"{age_val}대"

    logger.info(f"[NER] user_ner merged: {user_ner}")
    return user_ner

# ========================================================================
# (C) Tool 함수들: 프로필 업데이트, 벡터검색
# ========================================================================
def update_profile_tool_func(input_text: str) -> str:
    """
    LLM이 이 함수를 호출하면:
    1) _extract_user_ner()로 직무/지역/연령대 파악
    2) GLOBAL_USER_PROFILE 업데이트
    3) 현재 프로필 요약 리턴
    """
    user_ner = _extract_user_ner(input_text, GLOBAL_USER_PROFILE.data)

    # '직무' / '지역' / '연령대' 중 추출
    if "직무" in user_ner and user_ner["직무"]:
        jobs = GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])
        # 예: 여러 직무를 담고 싶다면 job.split() 등 가공
        if user_ner["직무"] not in jobs:
            jobs.append(user_ner["직무"])
        GLOBAL_USER_PROFILE.update("preferred_jobs", jobs)

    if "지역" in user_ner and user_ner["지역"]:
        GLOBAL_USER_PROFILE.update("location", user_ner["지역"])

    if "연령대" in user_ner and user_ner["연령대"]:
        # 숫자만 추출해서 age 업데이트도 가능
        # 여기서는 단순히 "60대" 같은 정보를 experience로 넣거나, 필요 시 별도 로직
        pass

    # 나이 같은 건 정규식으로 직접 추출할 수도 있음
    # 여기서는 생략

    return "프로필 업데이트 완료.\n" + GLOBAL_USER_PROFILE.get_profile_summary()

def vector_search_tool_func(_: str) -> str:
    """
    1) GLOBAL_USER_PROFILE에서 직무/지역 추출
    2) VectorStoreSearch 검색
    3) Document -> JobPosting 변환
    4) 최종 JSON 문자열로 반환
    """
    user_ner = {
        "직무": ",".join(GLOBAL_USER_PROFILE.data.get("preferred_jobs", [])),
        "지역": GLOBAL_USER_PROFILE.data.get("location", "")
    }

    docs = vector_store.search_jobs(user_ner, top_k=5)
    if not docs:
        return json.dumps({
            "jobPostings": [],
            "message": "조건에 맞는 채용 공고가 없습니다."
        }, ensure_ascii=False)

    # Document -> JobPosting
    job_postings = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        job_postings.append({
            "id": md.get("채용공고ID", "no_id"),
            "location": md.get("근무지역", ""),
            "company": md.get("회사명", ""),
            "title": md.get("채용제목", ""),
            "salary": md.get("급여조건", ""),
            "workingHours": md.get("근무시간", "정보없음"),
            "description": doc.page_content[:100],
            "rank": i
        })

    # JSON으로 반환
    return json.dumps({
        "jobPostings": job_postings,
        "message": "채용 공고 검색 결과"
    }, ensure_ascii=False)

# 1) 툴 정의
update_profile_tool = Tool(
    name="ProfileUpdateTool",
    func=update_profile_tool_func,
    description=(
        "사용자 메시지에서 나이나 직무, 지역 등의 정보를 추출하여 "
        "글로벌 사용자 프로필을 업데이트하는 도구."
    )
)

vector_search_tool = Tool(
    name="VectorSearchTool",
    func=vector_search_tool_func,
    description=(
        "사용자의 프로필(직무, 지역)에 맞춰 적합한 채용 공고를 벡터 스토어에서 검색하는 도구."
    )
)

# class JobAdvisorAgent:
#     """
#     고령자를 위한 취업 상담 에이전트.
#     벡터 검색(VectorStoreSearch)과 LLM(ChatOpenAI)을 연동하여,
#     사용자 프로필 기반 맞춤형 일자리를 추천한다.
#     (벡터 검색 + DB 기반 사용자 프로필 업데이트 + LLM 연동)
#     하나의 에이전트 객체: LLM + Tools
#     - 사용자가 "나 60세이고 서울에서 경비 알아봐"라고 하면:
#       -> LLM이 ProfileUpdateTool 호출 -> VectorSearchTool 호출 -> 최종 답변 
#     """

#     def __init__(self, openai_api_key: str = "", model_name: str = "gpt-4o-mini"):
#         self.llm = ChatOpenAI(
#             model=model_name,
#             temperature=0.5,
#             openai_api_key=openai_api_key
#         )
#         # Agent 초기화
#         self.tools = [update_profile_tool, vector_search_tool]

#         # system prompt
#         system_content = """
#         당신은 50세 이상 고령자의 취업을 지원하는 AI 상담사입니다.
#         사용자의 경험과 강점을 파악하여 맞춤형 일자리를 추천하고, 구직 활동을 지원합니다.

#         제공하는 주요 기능:
#         1. 경력/경험 기반 맞춤형 일자리 추천
#         2. 이력서 및 자기소개서 작성 가이드
#         3. 고령자 특화 취업 정보 제공
#         4. 면접 준비 및 커리어 상담
#         5. 디지털 취업 플랫폼 활용 방법 안내

#         상담 진행 방식:
#         1. 사용자의 기본 정보(나이, 경력, 희망 직종 등) 파악
#         2. 개인별 강점과 경험 분석
#         3. 맞춤형 일자리 정보 제공
#         4. 구체적인 취업 준비 지원

#         언제든 사용자의 메시지에서 나이, 직무, 지역 등을 추론하여 ProfileUpdateTool을 호출할 수 있고,
#         사용자 요구에 따라 VectorSearchTool을 사용해 채용공고를 검색할 수도 있습니다.

#         아래 규칙을 따르세요:
#         1. 쉽고 명확한 용어 사용
#         2. 프로필 업데이트나 채용 검색이 필요 없으면 툴을 호출하지 말고 직접 답변하세요.
#         3. 공감과 이해를 바탕으로 한 응대
#         4. 툴 호출 후에는 해당 결과를 바탕으로 최종 답변을 제시하세요.
#         5. 불필요한 사족은 달지 않는다. (예: 내부 Thought는 노출X)
#         """

#         self.agent_executor = initialize_agent(
#             tools=self.tools,
#             llm=self.llm,
#             agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#             verbose=True,  # 개발 중에 디버그 목적이라면 True로. 실제 서비스 땐 False로
#             system_message=SystemMessage(content=system_content)
#         )

#     def run(self, user_input: str) -> str:
#         """
#         사용자가 메시지를 보낼 때마다 run()을 호출.
#         에이전트는 필요한 도구(ProfileUpdate, VectorSearch)를 골라서 호출 후
#         최종 답변을 문자열로 반환한다.
#         """
#         logger.info(f"[JobAdvisorAgent] user_input: {user_input}")
#         answer = self.agent_executor.run(user_input)
#         return answer


###############################################################################
# (C) JobAdvisorAgent
###############################################################################
class JobAdvisorAgent:
    """
    고령자를 위한 취업 상담 에이전트 (ReAct).
    ReAct Agent가 ProfileUpdateTool, VectorSearchTool을 호출 가능.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        vector_search: VectorStoreSearch
    ):
        """
        llm: ChatOpenAI 인스턴스
        vector_search: 실제 벡터 검색 로직 (지금은 Tool 내부에서 더미)
        """
        self.llm = llm
        self.vector_search = vector_search

        # ReAct에 사용할 툴
        self.tools = [update_profile_tool, vector_search_tool]

        system_content = SYSTEM_PROMPT

        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            system_message=SystemMessage(content=system_content)
        )

    async def chat(self, query: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자가 보낸 메시지(query)를 토대로 ReAct Agent 실행.
        1) self.agent_executor.run(query) => 최종 답변(문자열)
        2) Tool(JSON) 결과가 섞여 있으면 파싱
        3) "message", "jobPostings", "type", "user_profile" 형태로 반환
        """
        logger.info(f"[JobAdvisorAgent] user_input: {query}")
        # (선택) user_profile을 GLOBAL_USER_PROFILE.data에 반영할 수도 있음
        # for k, v in user_profile.items():
        #     GLOBAL_USER_PROFILE.update(k, v)

        # ReAct 수행 (동기)
        # 만약 LangChain 0.0.202 이상이면 agent_executor.arun(...) 식으로 비동기 가능
        final_text = self.agent_executor.run(query)
        logger.info(f"[JobAdvisorAgent] ReAct final: {final_text}")

        # Tool이 반환한 JSON이 포함될 수도 있으므로 파싱 로직 (간단 예시)
        # ex) "채용공고 조회 결과: {\"jobPostings\": [...], \"message\":\"...\"}"
        job_postings = []
        msg_type = "info"

        # 정규식으로 {....} JSON 부분을 찾아 파싱 시도
        match = re.search(r"\{(\"jobPostings\".*?)\}$", final_text.strip())
        if match:
            raw_json = match.group(0)
            try:
                data = json.loads(raw_json)
                if "jobPostings" in data:
                    for i, jp in enumerate(data["jobPostings"], start=1):
                        jp["rank"] = i
                    job_postings = data["jobPostings"]
                    msg_type = "jobPosting"
                # 메시지 필드가 있으면 그걸로 교체
                final_text = data.get("message", final_text)
            except Exception as e:
                logger.warning(f"JSON parse fail in final_text: {e}")

        # 기본 반환
        return {
            "message": final_text,
            "jobPostings": job_postings,
            "type": msg_type,
            "user_profile": user_profile
        }