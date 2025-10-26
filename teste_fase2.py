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

base_dir = '/content/dataset2-master/dataset2-master/images/TEST'
saida_dir = '/content/dataset2-master/dataset2-master/images/VALIDAS'
nova_resolucao = (256, 256)
novo_formato = 'JPEG'
qualidade = 95
lote_plot = 5  # imagens por figura

os.makedirs(saida_dir, exist_ok=True) #cria pastas da saída de imagens, que se existirem o código continua normalmente

classes = [tipos for tipos in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, tipos))]
# cria uma pasta chamada classes, que tem somente dentro dela as pastas que tem sob a pasta principal.


# -------------------------------
# 1. Verificação de integridade
# -------------------------------
corrompidos = []

for classe in classes: #percorre cada pasta de classe dentro do dataset 
    caminho_classe = os.path.join(base_dir, classe) #monta o caminho que vai ser encontrado da classe
    for arquivo in os.listdir(caminho_classe):  #entra na classe e lista tudo o que tem dentro dela, arquivo por arquivo 
        caminho_arquivo = os.path.join(caminho_classe, arquivo) #junta o caminho e o nome do arquivo pra abrir a imagem
        try:
            with Image.open(caminho_arquivo) as img: #tenta abrir o arquivo como uma imagem pelo pillow e que vai ser automaticamente fechado depois
                img.verify() #verifica se a imagem não está corrompida
        except: #se for ocorrer erro, o código entra nesse bloco
            corrompidos.append(caminho_arquivo) #o joga para a lista de corrompidos, para uma pasta.

# -------------------------------
# 2. Verificação de baixa resolução (<64x64)
# -------------------------------
baixa_res = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        if caminho_arquivo in corrompidos:
            continue #se estiver em corrompidos ele já pula 
        try:
            with Image.open(caminho_arquivo) as img: #usa pillow pra abrir a imagem que está na local caminho arquivo #with abre temporariamente e fecha automaticamente
                if img.width < 64 or img.height < 64: #verifica o tamanho da imagem (resolução) 
                    baixa_res.append(caminho_arquivo) #se for baixa resolução, adiciona seu caminho a lista de baixa resolução 
        except:
            continue

# -------------------------------
# 3. Detectar duplicatas exatas
# -------------------------------
hashes = {}
duplicadas = []

for classe in classes:
    caminho_classe = os.path.join(base_dir, classe)
    for arquivo in os.listdir(caminho_classe):
        caminho_arquivo = os.path.join(caminho_classe, arquivo)
        if caminho_arquivo in corrompidos or caminho_arquivo in baixa_res:
            continue #ser for corrompido ou baixa resolução ele não vai entrar na busca já, diretamente
        try: #tenta executar o codigo de leitura e o hash do arquivo 
            with open(caminho_arquivo, 'rb') as f: #abre e le o arquivo como binário
                h = hashlib.sha256(f.read()).hexdigest() #cria um hash do arquivo, calcula o hash SHA-256, uma “impressão digital” única para aquele conteúdo, e transforma em um hexadecimal legível 
            if h in hashes: #verifica se o código já foi visto antes 
                duplicadas.append(caminho_arquivo) #se já existe, é uma imagem duplicada da outra e vai ser adicionado ou excluido depois.
            else:
                hashes[h] = caminho_arquivo #se não foi visto é adicionado ao dicionario como novo item. guarda o hash e o caminho.
        except: #se der errado entra aqui e passa pra próxima iteração do loop. 
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

