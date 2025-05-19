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
    sentimento_modelo = Column(String, nullable=False)  # Nova coluna
    data_criacao = Column(DateTime, default=datetime.utcnow)

# Deleta tabela antiga
Registro.__table__.drop(engine)

# Cria tabela nova
Base.metadata.create_all(bind=engine)

# Verifica e exclui registros (opcional)
db = SessionLocal()
db.query(Registro).delete()
db.commit()
db.close()