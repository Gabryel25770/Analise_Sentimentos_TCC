from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from googletrans import Translator # type: ignore
import os
from db_models import SessionLocal, Registro
from sqlalchemy import func, desc
from collections import defaultdict

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelo HuggingFace ---
modelos_huggingface = [
    "GABRYEL25770/TrainedModel"     # T5
    # ,"GABRYEL25770/DistilBERT_Model_TCC"  # DistilBERT
    # ,"GABRYEL25770/RoBERTa_Model_TCC"      # RoBERTa
]

tipos_modelos = [
    "t5",
    "bert",
    "bert"
]

tokenizers = []
models = []

for modelo_name, tipo in zip(modelos_huggingface, tipos_modelos):
    if tipo == "t5":
        tokenizer = T5Tokenizer.from_pretrained(modelo_name)
        model = T5ForConditionalGeneration.from_pretrained(modelo_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelo_name)
        model = AutoModelForSequenceClassification.from_pretrained(modelo_name)
    
    tokenizers.append(tokenizer)
    models.append(model.to(device))
    model.eval()

# --- Tradutor ---
translator = Translator()

# --- Funções ---
def traduzir_para_ingles(texto_pt):
    traducao = translator.translate(texto_pt, src='pt', dest='en')
    return traducao.text

def predict_sentiment(model, tokenizer, text, tipo_modelo):
    texto_en = traduzir_para_ingles(text)

    if tipo_modelo == "t5":
        input_text = f"classify sentiment: {texto_en}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

        if torch.cuda.is_available():
            model.half()

        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=20, num_beams=3)

        sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        inputs = tokenizer(texto_en, return_tensors="pt", max_length=512, truncation=True).to(device)

        if torch.cuda.is_available():
            model.half()

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        # Mapeia o id para o sentimento
        label_mapping = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        sentiment = label_mapping.get(predicted_class_id, "desconhecido")

    torch.cuda.empty_cache()
    return sentiment

def calcular_consenso(sentimentos):
    """Retorna o sentimento mais votado entre os três modelos"""
    from collections import Counter
    contagem = Counter(sentimentos)
    consenso = contagem.most_common(1)[0][0]
    return consenso


# def predict_sentiment(text):
#     texto_en = traduzir_para_ingles(text)

#     input_text = f"classify sentiment: {texto_en}"
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True).to(device)

#     model = T5ForConditionalGeneration.from_pretrained(modelo_huggingface).to(device)

#     if torch.cuda.is_available():
#         model.half()

#     model.eval()
#     with torch.no_grad():
#         outputs = model.generate(inputs["input_ids"], max_length=20, num_beams=3)

#     sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     del model
#     torch.cuda.empty_cache()

#     return sentiment

# --- Rota principal ---
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

    #consenso = calcular_consenso(resultados)

    return jsonify({
        #"consenso": consenso,
        "modelo_t5": resultados[0]  # T5
        #,"modelo_distilbert": resultados[1]  # DistilBERT
        #,"modelo_roberta": resultados[2]   # RoBERTa
    })

@app.route("/save", methods=["POST", "OPTIONS"])
def save():
    if request.method == "OPTIONS":
        return '', 200  # responde ao preflight
    
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

# --- Início ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
