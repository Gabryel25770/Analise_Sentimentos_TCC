from db_models import SessionLocal, Registro

def listar_registros():
    db = SessionLocal()
    resultados = db.query(Registro).all()

    for item in resultados:
        print(f"ID: {item.id} | Texto: {item.texto} | Sentimento: {item.sentimento} | Data: {item.data_criacao}")

    db.close()

if __name__ == "__main__":
    listar_registros()
