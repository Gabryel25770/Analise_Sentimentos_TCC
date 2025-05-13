from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator  # type: ignore
import os
from db_models import SessionLocal, Registro
from collections import Counter
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5500",
    "https://frontend-main-orcin.vercel.app"
]}}, supports_credentials=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelos_huggingface = ["GABRYEL25770/TrainedModel"]
tipos_modelos = ["t5"]

tokenizers = []
models = []

# Carrega modelo e aplica .half() se disponível
for modelo_name, tipo in zip(modelos_huggingface, tipos_modelos):
    if tipo == "t5":
        tokenizer = T5Tokenizer.from_pretrained(modelo_name)
        model = T5ForConditionalGeneration.from_pretrained(modelo_name)

        if torch.cuda.is_available():
            model = model.half()  # ou .to(torch.bfloat16) se sua GPU suportar melhor
        model = model.to(device)
        model.eval()

        tokenizers.append(tokenizer)
        models.append(model)

translator = Translator()

def traduzir_para_ingles(texto_pt):
    traducao = translator.translate(texto_pt, src='pt', dest='en')
    return traducao.text

def predict_sentiment(model, tokenizer, text, tipo_modelo):
    try:
        texto_en = traduzir_para_ingles(text)

        if tipo_modelo == "t5":
            input_text = f"classify sentiment: {texto_en}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=10,
                    num_beams=2
                )

            sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del inputs, input_ids, attention_mask, outputs
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return sentiment

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("[ERRO] GPU sem memória. Limpando cache.")
            if torch.cuda.is_available():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            return "erro_memoria"
        else:
            traceback.print_exc()
            return "erro_modelo"
    except Exception:
        traceback.print_exc()
        return "erro_modelo"

def calcular_consenso(sentimentos):
    contagem = Counter(sentimentos)
    consenso = contagem.most_common(1)[0][0]
    return consenso

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "Texto vazio"}), 400

    resultados = []

    for model, tokenizer, tipo in zip(models, tokenizers, tipos_modelos):
        sentimento = predict_sentiment(model, tokenizer, user_text, tipo)
        resultados.append(sentimento)

    return jsonify({
        "modelo_t5": resultados[0]
    })

@app.route("/save", methods=["POST", "OPTIONS"])
def save():
    if request.method == "OPTIONS":
        return '', 200

    data = request.json
    texto = data.get("text", "")
    sentimento = data.get("sentiment", "")

    if not texto or not sentimento:
        return jsonify({"error": "Texto e sentimento são obrigatórios."}), 400

    db = SessionLocal()
    novo_registro = Registro(texto=texto, sentimento=sentimento)
    db.add(novo_registro)
    db.commit()
    db.close()

    return jsonify({"message": "Registro salvo com sucesso!"})

@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    db = SessionLocal()
    
    # 1. Busca todos os registros
    registros = db.query(Registro).order_by(desc(Registro.data_criacao)).limit(20).all()

    # 2. Distribuição de sentimentos (contagem)
    sentimentos_contagem = db.query(
        Registro.sentimento, func.count(Registro.sentimento)
    ).group_by(Registro.sentimento).all()

    sentimentos = {
        "labels": [sentimento for sentimento, _ in sentimentos_contagem],
        "data": [count for _, count in sentimentos_contagem]
    }

    # 3. Análises por dia
    analises_por_dia = db.query(
        func.date(Registro.data_criacao), func.count(Registro.id)
    ).group_by(func.date(Registro.data_criacao)).order_by(func.date(Registro.data_criacao)).all()

    analises = {
        "labels": [str(data) for data, _ in analises_por_dia],
        "data": [count for _, count in analises_por_dia]
    }

    # 4. Registros para preencher a tabela
    registros_serializados = [{
        "id": r.id,
        "texto": r.texto,
        "sentimento": r.sentimento,
        "data_criacao": r.data_criacao.isoformat()
    } for r in registros]

    db.close()

    return jsonify({
        "sentimentos": sentimentos,
        "analisesPorDia": analises,
        "registros": registros_serializados
    })


# @app.route("/dashboard-data", methods=["GET"])
# def dashboard_data():
#     return jsonify({"message": "Rota ainda em construção."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)