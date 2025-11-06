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

def normalizacao(validas, saida_dir='/content/imagens_normalizadas'):
    """
    Aplica normalização de intensidade (0 a 1) nas imagens válidas e salva as novas versões.

    Parameters
    ----------
    validas : list[str]
        Lista de caminhos das imagens válidas.
    saida_dir : str
        Diretório de saída para salvar as imagens normalizadas.
    """

    os.makedirs(saida_dir, exist_ok=True)
    normalizadas = []

    for caminho in validas:
        try:
            # Lê a imagem
            img = cv2.imread(caminho)
            if img is None:
                continue

            # Aplica normalização de intensidade (para [0,1])
            img_norm = cv2.normalize(img, None, alpha=0, beta=1,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            # Converte de volta para 8 bits (0–255) para salvar
            img_saida = (img_norm * 255).astype('uint8')

            # Monta caminho de saída
            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_saida)
            normalizadas.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(normalizadas)}")
    return normalizadas

def remocao_ruido(normalizadas, saida_dir='/content/remocao_ruido'):

    os.makedirs(saida_dir, exist_ok=True)
    remocao_ruido = []

    for caminho in normalizadas:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            img_suavizada = cv2.GaussianBlur(img, (5,5), 0)

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_suavizada)
            remocao_ruido.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(remocao_ruido)}")
    return remocao_ruido 



def correcao_rgb(remocao_ruido, saida_dir='/content/correcao_cor'):

    os.makedirs(saida_dir, exist_ok=True)
    correcao_rgb = []

    for caminho in remocao_ruido:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            correcao_rgb.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(correcao_rgb)}")
    return correcao_rgb

def correcao_cinza(correcao_rgb, saida_dir='/content/correcao_cinza'):

    os.makedirs(saida_dir, exist_ok=True)
    correcao_cinza = []

    for caminho in correcao_rgb:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_cinza)
            correcao_cinza.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(correcao_cinza)}")
    return correcao_cinza

def desfoque(correcao_cinza, saida_dir='/content/desfoque'):

    os.makedirs(saida_dir, exist_ok=True)
    desfoque = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            img_desfocada = cv2.blur(img,(5,5))

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_desfocada)
            desfoque.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(desfoque)}")
    return desfoque

def rotacao(correcao_cinza, saida_dir='/content/rotacao'):

    os.makedirs(saida_dir, exist_ok=True)
    rotacao = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue

            width, height  = img.shape[:2]
            matriz_rotacao = cv2.getRotationMatrix2D((width/2, height/2), 45, 1.0)
            img_rotacionada = cv2.warpAffine(img, matriz_rotacao, (width, height))
            

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            cv2.imwrite(novo_caminho, img_rotacionada)
            rotacao.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(rotacao)}")
    return rotacao

def aumento_brilho(correcao_cinza, saida_dir='/content/aumento_brilho'):
    os.makedirs(saida_dir, exist_ok=True)
    aumento_brilho = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            cod_brilho = np.ones(img.shape, dtype="uint8") * 70
            img_brilho = cv2.add(img, cod_brilho)

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_brilho)
            aumento_brilho.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(aumento_brilho)}")
    return aumento_brilho

def diminuicao_brilho(correcao_cinza, saida_dir='/content/diminuicao_brilho'):
    os.makedirs(saida_dir, exist_ok=True)
    diminuicao_brilho = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            cod_brilho = np.ones(img.shape, dtype="uint8") * 70
            img_brilho_min = cv2.subtract(img, cod_brilho)

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_brilho_min)
            diminuicao_brilho.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(diminuicao_brilho)}")
    return diminuicao_brilho

def desfoque(correcao_cinza, saida_dir='/content/desfoque'):

    os.makedirs(saida_dir, exist_ok=True)
    desfoque = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            img_desfocada = cv2.blur(img,(5,5))

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            # Salva imagem normalizada
            cv2.imwrite(novo_caminho, img_desfocada)
            desfoque.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(desfoque)}")
    return desfoque

def inverter(correcao_cinza, saida_dir='/content/inverter'):

    os.makedirs(saida_dir, exist_ok=True)
    inverter = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue

            img_invertida = cv2.flip(img, -1) #(inverte nos dois eixos)
            

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            cv2.imwrite(novo_caminho, img_invertida)
            inverter.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(inverter)}")
    return inverter

def nitidez(correcao_cinza, saida_dir='/content/nitidez'):

    os.makedirs(saida_dir, exist_ok=True)
    nitidez = []

    for caminho in correcao_cinza:
        try:
            img = cv2.imread(caminho)
            if img is None:
                continue
            
            aumentando_nitidez = np.array([[-1,-1,-1],
                                           [-1, 10, -1],
                                           [-1,-1,-1]], dtype=np.float32)
            img_nitida = cv2.filter2D(img, -1, aumentando_nitidez) #(inverte nos dois eixos)
            

            nome_arquivo = os.path.basename(caminho)
            novo_caminho = os.path.join(saida_dir, nome_arquivo)

            cv2.imwrite(novo_caminho, img_nitida)
            nitidez.append(novo_caminho)

        except Exception as e:
            print(f"erro no {caminho}: {e}")

    print(f" total: {len(nitidez)}")
    return nitidez