from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Enum, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, date
import enum

Base = declarative_base()

class UserRole(enum.Enum):
    HR = "hr"
    USER = "user"

class EmploymentType(enum.Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"

class VacancyStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"

class InterviewStatus(enum.Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class ApplicationStatus(enum.Enum):
    PENDING = "pending"  # На рассмотрении
    INTERVIEW_SCHEDULED = "interview_scheduled"  # Интервью назначено
    INTERVIEW_COMPLETED = "interview_completed"  # Интервью пройдено
    ACCEPTED = "accepted"  # Принято
    REJECTED = "rejected"  # Отклонено

class ProcessingStatus(enum.Enum):
    PENDING = "pending"  # Ожидает обработки
    PROCESSING = "processing"  # В процессе обработки
    COMPLETED = "completed"  # Обработка завершена
    FAILED = "failed"  # Ошибка обработки

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.USER)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Profile fields
    first_name = Column(String)
    last_name = Column(String)
    phone = Column(String)
    birth_date = Column(Date)
    location = Column(String)  # Место жительства
    about = Column(Text)  # О себе
    desired_salary = Column(Integer)  # Желаемая зарплата
    ready_to_relocate = Column(Boolean, default=False)  # Готов к переезду
    employment_type = Column(Enum(EmploymentType))  # Тип занятости
    education = Column(JSON)  # Образование (массив объектов)
    skills = Column(JSON)  # Навыки (массив строк)
    work_experience = Column(JSON)  # Опыт работы (массив объектов)
    
    vacancies = relationship("Vacancy", back_populates="creator")
    resumes = relationship("Resume", back_populates="user")

class Vacancy(Base):
    __tablename__ = "vacancies"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(Text)
    salary_from = Column(Integer)
    salary_to = Column(Integer)
    location = Column(String)
    employment_type = Column(String)
    experience_level = Column(String)
    benefits = Column(Text, nullable=True)
    company = Column(String(255), nullable=True)  # Условия работы (через запятую)
    status = Column(Enum(VacancyStatus), default=VacancyStatus.OPEN)
    original_url = Column(String)
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    creator = relationship("User", back_populates="vacancies")
    resumes = relationship("Resume", back_populates="vacancy")
    interviews = relationship("Interview", back_populates="vacancy")

    auto_interview_enabled = Column(Boolean, default=False)
    # Порог автоматического назначения интервью (0-100)
    auto_interview_threshold = Column(Integer)

class Resume(Base):
    __tablename__ = "resumes"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    vacancy_id = Column(Integer, ForeignKey("vacancies.id"))
    file_path = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    uploaded_by_hr = Column(Boolean, default=False)
    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.PENDING)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    notes = Column(Text)  # Заметки HR
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    hidden_for_hr = Column(Boolean, default=False)  # Soft-delete для отображения у HR
    
    user = relationship("User", back_populates="resumes")
    vacancy = relationship("Vacancy", back_populates="resumes")
    analysis = relationship("ResumeAnalysis", back_populates="resume", uselist=False)
    interviews = relationship("Interview", back_populates="resume")

class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), unique=True)
    
    name = Column(String)
    position = Column(String)
    experience = Column(String)
    education = Column(String)
    upload_date = Column(String)
    match_score = Column(String)
    key_skills = Column(JSON)
    recommendation = Column(String)
    
    projects = Column(JSON)
    work_experience = Column(JSON)
    technologies = Column(JSON)
    achievements = Column(JSON)
    
    # Новые поля для стабильно отображаемого анализа
    strengths = Column(JSON)  # Сильные стороны (массив строк)
    weaknesses = Column(JSON)  # Слабые стороны (массив строк)
    missing_skills = Column(JSON)  # Отсутствующие навыки (массив строк)
    brief_reason = Column(Text)  # Краткое объяснение вердикта
    
    structured = Column(Boolean)
    effort_level = Column(String)
    
    suspicious_phrases_found = Column(Boolean)
    suspicious_examples = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    resume = relationship("Resume", back_populates="analysis")

class Interview(Base):
    __tablename__ = "interviews"
    
    id = Column(Integer, primary_key=True, index=True)
    vacancy_id = Column(Integer, ForeignKey("vacancies.id"))
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    status = Column(Enum(InterviewStatus), default=InterviewStatus.NOT_STARTED)
    
    scheduled_date = Column(DateTime)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    duration_minutes = Column(Integer)
    
    dialogue = Column(JSON)
    summary = Column(Text)
    pass_percentage = Column(Float)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    vacancy = relationship("Vacancy", back_populates="interviews")
    resume = relationship("Resume", back_populates="interviews")
