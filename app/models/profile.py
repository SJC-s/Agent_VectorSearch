# app/models/profile.py
from datetime import datetime

class UserProfile:
    """
    고령자 사용자의 나이, 경력, 선호 직종, 기술, 위치, 학력, 상태 등을
    저장 및 관리하는 클래스
    """
    def __init__(self):
        self.data = {
            "age": None,
            "experience": [],
            "preferred_jobs": [],
            "skills": [],
            "location": None,
            "education": None,
            "job_status": None
        }
        self.conversation_state = "initial"
        self.last_update = datetime.now()

    def update(self, key: str, value: any):
        self.data[key] = value
        self.last_update = datetime.now()

    def get_profile_summary(self) -> str:
        if not self.data["age"]:
            return "프로필 정보가 아직 없습니다."
        
        summary = f"""현재 프로필 정보:
- 나이: {self.data['age']}세
- 직무: {', '.join(self.data['preferred_jobs']) if self.data['preferred_jobs'] else '미입력'}
- 경력: {', '.join(self.data['experience']) if self.data['experience'] else '미입력'}
- 보유 기술: {', '.join(self.data['skills']) if self.data['skills'] else '미입력'}
- 지역: {self.data['location'] if self.data['location'] else '미입력'}
- 학력: {self.data['education'] if self.data['education'] else '미입력'}
- 현재 상태: {self.data['job_status'] if self.data['job_status'] else '미입력'}"""
        return summary
