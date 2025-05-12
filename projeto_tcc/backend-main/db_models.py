from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import DateTime


# --- Conexão com SQLite ---
DATABASE_URL = "sqlite:///textos.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# --- Modelo ---
Base = declarative_base()

class Registro(Base):
    __tablename__ = "registros"
    id = Column(Integer, primary_key=True, index=True)
    texto = Column(String, nullable=False)
    sentimento = Column(String, nullable=False)
    data_criacao = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)