import os
import shutil
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Diretórios
ROOT_DIR = r'C:\Users\donza\OneDrive\Python\BaixarFotoFolia\fotos_fotofolia'
TREINO_DIR = os.path.join(ROOT_DIR, 'treino')
METAIS_DIR = os.path.join(ROOT_DIR, 'Metais')
os.makedirs(METAIS_DIR, exist_ok=True)

# Parâmetros
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset completo
full_dataset = datasets.ImageFolder(TREINO_DIR, transform=transform)
class_names = full_dataset.classes
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

# Divisão treino/validação
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Modelo
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# Treinamento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("🧪 Treinando modelo...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"  🔁 Loss da época {epoch+1}: {running_loss:.4f}")

# Avaliação
print("\n📊 Avaliando modelo...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Relatório
print("\n🔬 Relatório de Classificação:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.tight_layout()
#plt.show()

# Salvar modelo
modelo_path = os.path.join(ROOT_DIR, "modelo_metais.pth")
torch.save(model.state_dict(), modelo_path)
print(f"✅ Modelo salvo em: {modelo_path}")

# Função de predição
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(probs, 1)
        label = idx_to_class[predicted.item()]
        conf = probs[0][predicted.item()].item()
    return label, conf

# Classificação de novas imagens
print("\n🔍 Classificando novas imagens...\n")
for root, dirs, files in os.walk(ROOT_DIR):
    if 'treino' in root or 'Metais' in root:
        continue
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            print(f"\n📁 Pasta: {folder_name}")
            print(f"🖼️  Arquivo: {file}")
            try:
                label, conf = predict_image(image_path)
                print(f"🔍 Resultado: {label} ({conf:.2f})")
                if label == 'metais' and conf > 0.8:
                    new_name = f"{folder_name}_{file}"
                    destino = os.path.join(METAIS_DIR, new_name)
                    shutil.move(image_path, destino)
                    print(f"✅ [METAIS] Movido para: {new_name}")
                else:
                    print(f"➖ [OUTRO] Arquivo mantido.")
            except Exception as e:
                print(f"⚠️ Erro com {file}: {e}")
