from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator

app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "*"}})  # Libera acesso para o frontend se comunicar com o backend

# Carregar modelo diretamente do Hugging Face Hub
modelo_huggingface = "GABRYEL25770/TrainedModel"

tokenizer = T5Tokenizer.from_pretrained(modelo_huggingface)
model = T5ForConditionalGeneration.from_pretrained(modelo_huggingface)

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inicializa tradutor
translator = Translator()

def traduzir_para_ingles(texto_pt):
    """Traduz do português para o inglês usando Google Translate."""
    traducao = translator.translate(texto_pt, src='pt', dest='en')
    return traducao.text

def predict_sentiment(text):
    """Traduz a frase para inglês e faz a inferência com T5."""
    texto_en = traduzir_para_ingles(text)

    input_text = f"classify sentiment: {texto_en}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"])

    sentiment = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sentiment

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "Texto vazio"}), 400

    sentiment = predict_sentiment(user_text)
    
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

