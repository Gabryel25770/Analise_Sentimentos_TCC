from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator  # type: ignore
import os
from db_models import SessionLocal, Registro
from collections import Counter

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
            model = model.half()
        
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
            inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=10,  # REDUZIDO para economizar memória
                    num_beams=2     # REDUZIDO para economizar memória
                )

            sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del inputs
        del outputs
        torch.cuda.empty_cache()

        return sentiment

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("[ERRO] Memória insuficiente na GPU. Forçando limpeza e fallback.")
            torch.cuda.empty_cache()
            return "erro_memoria"
        raise e  # outros erros continuam sendo lançados

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
    return jsonify({"message": "Rota ainda em construção."})
