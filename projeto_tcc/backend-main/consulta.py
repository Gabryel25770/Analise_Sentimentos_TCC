from db_models import SessionLocal, Registro

db = SessionLocal()

registros = db.query(Registro).all()

for r in registros:
    print(f"ID: {r.id}, Texto: {r.texto}, Sentimento: {r.sentimento}, Modelo: {r.sentimento_modelo}, Data: {r.data_criacao}")

db.close()