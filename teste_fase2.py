!pip install kaggle 

from google.colab import drive
drive.mount('/content/drive')

! mkdir ~/.kaggle

! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download paultimothymooney/blood-cells

! unzip blood-cells.zip

import os
from PIL import Image
import hashlib
import matplotlib.pyplot as plt
import math

# Configurações

base_dir = '/content/dataset2-master/dataset2-master/images/TEST'
saida_dir = '/content/dataset2-master/dataset2-master/images/VALIDAS'
nova_resolucao = (256, 256)
novo_formato = 'JPEG'
qualidade = 95
lote_plot = 5  # imagens por figura

# Criar pasta de saída se não existir
os.makedirs(saida_dir, exist_ok=True)

classes = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, c))]

# 1. Verificação de integridade

corrompidos = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        try:
            with Image.open(caminho_arquivo) as img:
                img.verify()
        except:
            corrompidos.append(caminho_arquivo)


# 2. Verificação de baixa resolução (<64x64)

baixa_res = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        if caminho_arquivo in corrompidos:
            continue
        try:
            with Image.open(caminho_arquivo) as img:
                if img.width < 64 or img.height < 64:
                    baixa_res.append(caminho_arquivo)
        except:
            continue

# 3. Detectar duplicatas exatas

hashes = {}
duplicadas = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        if caminho_arquivo in corrompidos or caminho_arquivo in baixa_res:
            continue
        try:
            with open(caminho_arquivo, 'rb') as f:
                h = hashlib.sha256(f.read()).hexdigest()
            if h in hashes:
                duplicadas.append(caminho_arquivo)
            else:
                hashes[h] = caminho_arquivo
        except:
            continue


# 4. Padronizar imagens válidas e salvar

validas = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    nova_classe_dir = os.path.join(saida_dir, classe)
    os.makedirs(nova_classe_dir, exist_ok=True)
    
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        if caminho_arquivo in corrompidos or caminho_arquivo in baixa_res or caminho_arquivo in duplicadas:
            continue
        try:
            with Image.open(caminho_arquivo) as img:
                img = img.convert('RGB')
                img = img.resize(nova_resolucao, Image.Resampling.LANCZOS)
                novo_nome = os.path.join(nova_classe_dir, os.path.splitext(arquivo)[0] + '.jpg')
                img.save(novo_nome, novo_formato, quality=qualidade)
                validas.append(novo_nome)
        except Exception as e:
            print(f"Erro em {caminho_arquivo}: {e}")

print(f"Total de imagens válidas e padronizadas: {len(validas)}")

# 5. Plotar imagens válidas por classe (lotes de 5)

for classe in classes:
    imagens_classe = [img for img in validas if f'/{classe}/' in img]
    if not imagens_classe:
        continue
    
    for i in range(0, len(imagens_classe), lote_plot):
        lote = imagens_classe[i:i+lote_plot]
        plt.figure(figsize=(15, 5*len(lote)))  # altura proporcional ao lote
        colunas = 1
        linhas = len(lote)
        plt.suptitle(f"Imagens válidas da classe {classe}", fontsize=16)
        
        for j, img_path in enumerate(lote):
            img = Image.open(img_path)
            plt.subplot(linhas, colunas, j+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path), fontsize=12)
        
        plt.tight_layout(pad=3.0)
        plt.show()

