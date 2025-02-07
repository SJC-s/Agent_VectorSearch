## 1. 디렉토리 구조

```plaintext
FastApi_SeniorJobGo/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat_router.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_store_ingest.py        # 문서 인덱싱 (저장) 관련 모듈
│   │   ├── vector_store_search.py        # 문서 검색 (유사도 검색) 관련 모듈
│   │   └── conversation.py               # 대화 처리 비즈니스 로직
│   ├── agents/
│   │   ├── __init__.py
│   │   └── job_advisor.py                # LangGraph 에이전트 (도구 포함)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py                    # Pydantic 모델
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                     # 환경 변수 & 설정
│   │   └── prompts.py                    # 시스템 프롬프트, 템플릿 등
│   └── utils/
│       ├── __init__.py
│       └── constants.py                  # 상수/키워드 정의
├── documents/
│   └── jobs.json                         # 샘플 채용 정보
├── jobs_collection/                      # 벡터 DB 파일(Chroma 등) 저장 경로
├── requirements.txt
├── .env
└── README.md
```

vector_store_ingest.py: 문서를 벡터 DB에 인덱싱(저장)하는 기능.
vector_store_search.py: 쿼리를 받아 벡터 DB 유사도 검색을 수행하는 기능.
job_advisor.py: LangGraph 에이전트 정의. 여기서 위 두 모듈을 도구(Tool) 형태로 등록하여, AI가 필요 시 호출.

### 디렉토리 설명
1. **API 라우팅** (`routes/`), 
2. **비즈니스 로직** (`services/`), 
3. **LangChain/LangGraph 기반 AI 에이전트** (`agents/`), 
4. **데이터 모델** (`models/`), 
5. **환경설정** (`core/`, `.env`), 
6. **유틸** (`utils/`)  
등이 **명료하게 분리**되어 유지보수와 확장성이 높아진다. 
