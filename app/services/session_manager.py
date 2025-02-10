# /app/services/session_manager.py
from sqlalchemy import text
from typing import List, Dict

class ChatSession:
    """
    DB를 사용할 때는 실제로는 이 객체가 큰 의미는 없을 수 있다.
    사용자 프로필만 메모리나 DB에 따로 저장하고,
    메시지는 DB에서 바로 조회하는 방식을 사용한다.
    """
    def __init__(self):
        self.user_profile: Dict = {
            "age": None,
            "location": None,
            "jobType": None,
            "experience": [],
            "preferred_jobs": [],
            "skills": [],
            "education": None,
            "job_status": None
        }

class SessionManager:
    def __init__(self):
        # self.sessions = {}  # 기존 in-memory 방식을 사용하지 않음
        pass
    
    # def get_or_create_session(self, session_id: str) -> ChatSession:
    #     """
    #     DB에서 session_id와 관련된 프로필 정보를 조회하여
    #     ChatSession 객체로 만들어 반환.
    #     """
    #     # 예: 프로필을 저장하는 별도 테이블이 있다고 가정
    #     # 여기서는 단순히 ChatSession 인스턴스를 리턴한다고 가정
    #     session = ChatSession()
    #     # DB에서 사용자 정보를 읽어 세션에 채울 수도 있다.
    #     return session
    
    # def update_session(self, session_id: str, message: Dict, user_profile: Dict = None):
    #     """
    #     message를 DB 테이블(conversation_logs)에 저장한다.
    #     user_profile이 있으면 DB나 세션에 업데이트한다.
    #     """
    #     query = text("""
    #         INSERT INTO conversation_logs (session_id, role, content)
    #         VALUES (:session_id, :role, :content)
    #     """)
    #     with engine.connect() as conn:
    #         conn.execute(query, {
    #             "session_id": session_id,
    #             "role": message["role"],
    #             "content": message["content"]
    #         })
    #         conn.commit()
        
    #     # 유저 프로필도 DB에 따로 저장하는 방식을 사용할 수 있다.
    #     if user_profile:
    #         # 예: 별도 user_profile 테이블이 있다면
    #         # update user_profile set ... where session_id = :session_id
    #         # 또는 JSON 컬럼을 사용해 user_profile을 그대로 저장하는 방법도 있음
    #         pass
    
    # def get_messages(self, session_id: str) -> List[Dict]:
    #     """
    #     session_id에 해당하는 모든 메시지를 DB에서 불러온다.
    #     """
    #     query = text("""
    #         SELECT role, content
    #         FROM conversation_logs
    #         WHERE session_id = :session_id
    #         ORDER BY created_at ASC
    #     """)
    #     with engine.connect() as conn:
    #         rows = conn.execute(query, {"session_id": session_id}).fetchall()
        
    #     messages = []
    #     for row in rows:
    #         messages.append({"role": row.role, "content": row.content})
    #     return messages
    
    # def get_user_profile(self, session_id: str) -> Dict:
    #     """
    #     세션별 유저 프로필을 DB에서 가져와서 반환한다.
    #     실제 구현에서는 user_profile 테이블을 만들어 session_id와 1:1로 매핑 가능.
    #     """
    #     session = self.get_or_create_session(session_id)
    #     return session.user_profile
