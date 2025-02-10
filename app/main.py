# app/main.py
import uvicorn
import signal
import sys
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes.chat_router import router as chat_router
from app.agents.job_advisor import JobAdvisorAgent
from app.services.vector_store_search import VectorStoreSearch
# from app.services.session_manager import session_manager  # DB 세션 로직
from app.services.vector_store_ingest import VectorStoreIngest

from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="SeniorJobGo",
        description="시니어 취업 지원 플랫폼 (LangGraph Agent)",
        version="1.0.0"
    )

    # CORS 설정
    origins = [
        "http://localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        # allow_origins=origins,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터
    app.include_router(chat_router, prefix="/api/v1")

    return app

app = create_app()
vector_store = None

# @asynccontextmanager
# async def startup_event():
#     """
#     서버 시작 시점에 다음을 초기화:
#     1) 벡터 스토어(Chroma) - 인덱싱 or 검색 전용
#     2) LLM
#     3) JobAdvisorAgent
#     4) app.state에 저장
#     """
#     logger.info("[Startup] 벡터 스토어를 초기화합니다.")
#     # 1) 벡터 스토어(ingest or search)
#     # (예: 인덱싱 전용 객체)
#     ingest = VectorStoreIngest() 
    
#     # 검색 전용
#     vector_search_obj = VectorStoreSearch(ingest.setup_vector_store())  # 내부에서 setup
#     logger.info("[Startup] vector_search_obj 로딩 완료.")

#     # 2) LLM 준비
#     llm = ChatOpenAI(
#         model="gpt-4o-mini",
#         temperature=0.5,
#         streaming=True
#     )
#     logger.info("[Startup] LLM 초기화 완료.")

#     # 3) JobAdvisorAgent
#     job_advisor = JobAdvisorAgent(
#         llm=llm,
#         vector_search=vector_search_obj,
#         # session_manager=session_manager,  # DB 세션
#     )
#     logger.info("[Startup] JobAdvisorAgent 생성 완료.")

#     # 4) app.state 등록
#     app.state.job_advisor_agent = job_advisor
#     logger.info("[Startup] 에이전트 등록 완료.")

@app.on_event("startup")
def on_startup():
    """
    서버 시작 시점에 다음을 초기화:
    1) 벡터 스토어(Chroma)
    2) LLM
    3) JobAdvisorAgent
    4) app.state에 저장
    """
    logger.info("[Startup] 벡터 스토어를 초기화합니다.")
    ingest = VectorStoreIngest()
    vector_search_obj = VectorStoreSearch(ingest.setup_vector_store())  # 내부적으로 Chroma 로드

    logger.info("[Startup] LLM 초기화.")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        streaming=False
    )

    logger.info("[Startup] JobAdvisorAgent 생성.")
    job_advisor = JobAdvisorAgent(
        llm=llm,
        vector_search=vector_search_obj
    )

    app.state.job_advisor_agent = job_advisor
    logger.info("[Startup] 에이전트 등록 완료.")

def signal_handler(sig, frame):
    logger.info(f"시그널 {sig} 감지. 서버를 안전하게 종료합니다...")
    sys.exit(0)

if __name__ == "__main__":
    # 시그널 핸들러
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
