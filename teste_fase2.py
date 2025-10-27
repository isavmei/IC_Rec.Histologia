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


def obter_corrompidos(base_dir: str, classes: list[str]) -> list[str]:
  """
  Verifica imagens corrompidas.

  O script percorre cada pasta da classe, lista seus arquivos de imagem e
  tenta abrir cada imagem individualmente. Se não for possível abrir a 
  imagem, ela será considerada corrompida, e será retornada.

  Parameters
  ----------
  base_dir : str
    Caminho para o diretório base.

  classes : list[str]
    Lista de classes possíveis.

  Returns
  -------
  list[str]
    Lista de caminhos para arquivos corrompidos. Caso nenhuma imagem esteja corrompida, será
    retornada uma lista vazia.
  """
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
  return corrompidos
# -------------------------------
# 2. Verificação de baixa resolução (<64x64)
# -------------------------------

import os
from PIL import Image

def obter_baixa_resolucao(base_dir: str, classes: list[str], corrompidos: list[str]) -> list[str]:
    """
    Verifica imagens com baixa resolução.
    """
    baixa_res = []

    for classe in classes:
        caminho_classe = os.path.join(base_dir, classe)
        for arquivo in os.listdir(caminho_classe):
            caminho_arquivo = os.path.join(caminho_classe, arquivo)
            if caminho_arquivo in corrompidos:
                continue  # se estiver em corrompidos ele já pula 
            try:
                with Image.open(caminho_arquivo) as img:
                    if img.width < 64 or img.height < 64:
                        baixa_res.append(caminho_arquivo)
            except:
                continue

    return baixa_res


def detectar_duplicatas(base_dir: str, classes: list[str], corrompidos: list[str], baixa_res: list[str]) -> list[str]:
    """
    Detecta imagens duplicadas exatas dentro das classes, ignorando arquivos corrompidos ou de baixa resolução.
    
    Parameters
    ----------
    base_dir : str
        Caminho para o diretório base.
    classes : list[str]
        Lista de classes possíveis.
    corrompidos : list[str]
        Lista de caminhos de arquivos corrompidos.
    baixa_res : list[str]
        Lista de caminhos de arquivos de baixa resolução.
    
    Returns
    -------
    list[str]
        Lista de caminhos de arquivos duplicados.
    """
    hashes = {}       # Dicionário para guardar hash de cada arquivo
    duplicadas = []   # Lista de arquivos duplicados encontrados

    for classe in classes:
        caminho_classe = os.path.join(base_dir, classe)
        for arquivo in os.listdir(caminho_classe):
            caminho_arquivo = os.path.join(caminho_classe, arquivo)

            # Ignora arquivos corrompidos ou de baixa resolução
            if caminho_arquivo in corrompidos or caminho_arquivo in baixa_res:
                continue

            try:
                # Abre o arquivo em modo binário e calcula o hash SHA-256
                with open(caminho_arquivo, 'rb') as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                
                # Verifica se o hash já foi visto
                if h in hashes:
                    duplicadas.append(caminho_arquivo)  # Arquivo duplicado
                else:
                    hashes[h] = caminho_arquivo  # Guarda o hash como visto
            except:
                continue  # Se der erro, apenas ignora e segue

    return duplicadas

def padronizar_imagens(base_dir: str, classes: list[str], saida_dir: str,
                       corrompidos: list[str], baixa_res: list[str], duplicadas: list[str],
                       nova_resolucao: tuple[int, int], novo_formato: str = 'JPEG', qualidade: int = 95) -> list[str]:
    """
    Converte e redimensiona imagens válidas, salvando-as em novo diretório.

    Parameters
    ----------
    base_dir : str
        Diretório base com as classes originais.
    classes : list[str]
        Lista de classes a serem processadas.
    saida_dir : str
        Diretório de saída onde as imagens padronizadas serão salvas.
    corrompidos : list[str]
        Lista de arquivos corrompidos.
    baixa_res : list[str]
        Lista de arquivos de baixa resolução.
    duplicadas : list[str]
        Lista de arquivos duplicados.
    nova_resolucao : tuple[int, int]
        Nova resolução (largura, altura) para as imagens.
    novo_formato : str, opcional
        Formato de saída das imagens (padrão 'JPEG').
    qualidade : int, opcional
        Qualidade da imagem salva (padrão 95).

    Returns
    -------
    list[str]
        Lista de caminhos das imagens válidas processadas.
    """
    validas = []

    for classe in classes:
        caminho_classe = os.path.join(base_dir, classe)
        nova_classe_dir = os.path.join(saida_dir, classe)
        os.makedirs(nova_classe_dir, exist_ok=True)  # Cria diretório se não existir

        for arquivo in os.listdir(caminho_classe):
            caminho_arquivo = os.path.join(caminho_classe, arquivo)

            # Ignora arquivos corrompidos, de baixa resolução ou duplicados
            if caminho_arquivo in corrompidos or caminho_arquivo in baixa_res or caminho_arquivo in duplicadas:
                continue

            try:
                # Abre a imagem, converte para RGB e redimensiona
                with Image.open(caminho_arquivo) as img:
                    img = img.convert('RGB')
                    img = img.resize(nova_resolucao, Image.Resampling.LANCZOS)

                    # Salva no novo diretório com o novo formato
                    novo_nome = os.path.join(nova_classe_dir, os.path.splitext(arquivo)[0] + '.jpg')
                    img.save(novo_nome, novo_formato, quality=qualidade)

                    validas.append(novo_nome)  # Adiciona à lista de válidas

            except Exception as e:
                print(f"Erro em {caminho_arquivo}: {e}")

    print(f"Total de imagens válidas e padronizadas: {len(validas)}")
    return validas

