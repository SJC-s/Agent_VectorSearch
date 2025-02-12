# app/main.py
import uvicorn
import signal
import sys
import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.core.config import settings

from langchain_openai import ChatOpenAI
from app.services.vector_store_search import VectorStoreSearch
from app.services.vector_store_ingest import VectorStoreIngest
from app.agents.job_advisor import build_job_advisor_graph
from app.routes import chat_router
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        logger.info("벡터 스토어를 초기화합니다. (ingest)")
        ingest = VectorStoreIngest()  # DB 생성/로드 담당
        collection = ingest.setup_vector_store()  # Chroma 객체
        
        logger.info("벡터 스토어 검색 객체를 초기화합니다. (search)")
        vector_search = VectorStoreSearch(collection)
        app.state.vector_search = vector_search  # vector_search를 app.state에 저장
        
        llm_instance = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name="gpt-4o-mini",
            temperature=0.5
        )

        global graph
        graph = build_job_advisor_graph(llm=llm_instance)
        app.state.graph = graph
        logger.info("초기화 완료")
        
        
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield  # lifespan 종료 시점

    # shutdown
    logger.info("서버를 종료합니다...")

# FastAPI 앱 생성 시 lifespan 설정
app = FastAPI(lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
vector_store = None
job_advisor_agent = None
llm = None

app.include_router(chat_router.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)