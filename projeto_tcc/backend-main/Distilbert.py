import torch
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.optim import AdamW  # AdamW é recomendado para Transformers
import pandas as pd
from collections import Counter

# Carregar o dataset
df = pd.read_csv('dataset_pronto.csv', delimiter=';')

# Remover espaços extras e valores nulos
df['Classification'] = df['Classification'].astype(str).str.strip()
df = df.dropna(subset=['Phrase'])

# Filtrar rótulos válidos
valid_labels = {'positive', 'negative', 'neutral'}
df = df[df['Classification'].isin(valid_labels)]

# Mapear rótulos para números
label_map = {'positive': 1, 'negative': 0, 'neutral': 2}
df['Classification'] = df['Classification'].map(label_map)

# Extrair textos e rótulos
texts = df['Phrase'].tolist()
labels = df['Classification'].tolist()

# Verificar distribuição das classes
label_counts = Counter(labels)
print("Distribuição das classes:", label_counts)

# Definição do dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Carregar o tokenizer e o modelo DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Separação dos dados em treino e teste
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Criar os datasets
dataset_train = SentimentDataset(train_texts, train_labels, tokenizer)
dataset_val = SentimentDataset(val_texts, val_labels, tokenizer)

# Criar os DataLoaders
batch_size = 8
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# Criar pesos para lidar com desbalanceamento de classes
total_samples = sum(label_counts.values())
class_weights = torch.tensor(
    [total_samples / (len(label_counts) * label_counts[i]) if i in label_counts else 1.0 for i in range(3)],
    dtype=torch.float
)
class_weights = class_weights / class_weights.sum()

# Enviar modelo para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights = class_weights.to(device)

print("Dispositivo em uso:", device)

# Otimizador e função de perda
optimizer = AdamW(model.parameters(), lr=3e-5)
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

# Função de treinamento
def train(model, dataloader, optimizer, loss_function):
    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
        loss = loss_function(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Batch {i}/{len(dataloader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss  # Agora retorna o loss médio

# Função de avaliação
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    accuracy = correct / total if total > 0 else 0  # Evita divisão por zero
    return accuracy  # Agora retorna a acurácia

# Listas para armazenar os valores de loss e acurácia por época
loss_values = []
accuracy_values = []

# Treinar e avaliar o modelo
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    avg_loss = train(model, dataloader_train, optimizer, loss_function)
    accuracy = evaluate(model, dataloader_val)
    
    # Armazenar os valores para o gráfico
    loss_values.append(avg_loss)
    accuracy_values.append(accuracy)

    print(f"Loss médio: {avg_loss:.4f}, Acurácia: {accuracy:.4f}")

# Função para testar uma frase personalizada
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    inverse_label_map = {v: k for k, v in label_map.items()}
    return inverse_label_map[prediction]

# Teste com frase personalizada
input_text = "I really love this product! It's amazing."
sentiment = predict_sentiment(input_text, model, tokenizer, device)
print(f"Sentimento previsto: {sentiment}")

# Gerar gráfico de loss e acurácia
plt.figure(figsize=(10,5))

# Plotar loss médio
plt.plot(range(1, num_epochs+1), loss_values, label='Loss Médio', marker='o')

# Plotar acurácia
plt.plot(range(1, num_epochs+1), accuracy_values, label='Acurácia', marker='s')

# Configurações do gráfico
plt.xlabel('Épocas')
plt.ylabel('Valores')
plt.title('Evolução do Loss e Acurácia durante o Treinamento')
plt.legend()
plt.grid()
plt.show()
