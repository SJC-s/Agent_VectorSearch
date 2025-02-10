# app/main.py
import uvicorn
import signal
import sys
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

load_dotenv()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        logger.info("벡터 스토어를 초기화합니다. (ingest)")
        ingest = VectorStoreIngest()  # DB 생성/로드 담당
        collection = ingest.setup_vector_store()  # Chroma 객체
        
        logger.info("벡터 스토어 검색 객체를 초기화합니다. (search)")
        vector_search = VectorStoreSearch(collection)
        
        logger.info("LLM과 에이전트를 초기화합니다.")
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        
        app.state.graph = build_job_advisor_graph(
            llm=llm,
            vector_search=vector_search  # 검색 전용 객체 주입
        )
        logger.info("초기화 완료")
        
        
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        raise
        
    yield
    
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

app.include_router(chat_router.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
