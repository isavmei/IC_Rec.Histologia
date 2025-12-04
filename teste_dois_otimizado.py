!pip install kaggle tqdm --quiet

from google.colab import drive
drive.mount('/content/drive')

! mkdir ~/.kaggle

! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download paultimothymooney/blood-cells

! unzip blood-cells.zip

import os
from pathlib import Path
import shutil
import hashlib
import cv2
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==========================
# CONFIGURAÇÃO
# ==========================

BASE_DIR = Path('/content/dataset2-master/dataset2-master/images/TEST')
OUT_BASE = Path('/content/dataset2-master/dataset2-master/images/RESULTADOS')

DIR_CORR = OUT_BASE / "CORROMPIDAS"
DIR_LOW  = OUT_BASE / "BAIXA_RES"
DIR_DUP  = OUT_BASE / "DUPLICADAS"
DIR_OK   = OUT_BASE / "MELHORADAS"
DIR_GRAY = OUT_BASE / "CINZA"

for d in [DIR_CORR, DIR_LOW, DIR_DUP, DIR_OK, DIR_GRAY]:
    d.mkdir(parents=True, exist_ok=True)

N_WORKERS = max(1, os.cpu_count() - 1)
TARGET_RES = (256, 256)
JPEG_QUALITY = 95
MIN_DIM = 64


# ==========================
# UTILITÁRIOS
# ==========================

def sha256_of_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def list_all_image_files(base_dir: Path):
    files = []
    for cls in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        for f in cls.iterdir():
            if f.is_file():
                files.append((cls.name, f))
    return files


# ==========================
# INSPEÇÃO
# ==========================

def inspect_file(tuple_cls_path):
    cls, path = tuple_cls_path
    path = Path(path)
    try:
        b = path.read_bytes()
        if len(b) == 0:
            return (str(path), cls, False, None, None)

        h = sha256_of_bytes(b)

        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return (str(path), cls, False, None, h)

        if img.ndim == 2:
            dims = (img.shape[1], img.shape[0])
        else:
            dims = (img.shape[1], img.shape[0])

        return (str(path), cls, True, dims, h)

    except:
        return (str(path), cls, False, None, None)


def run_inspection(files):
    meta = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = [exe.submit(inspect_file, f) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Inspecionando"):
            meta.append(fut.result())
    return meta


# ==========================
# CLASSIFICAÇÃO
# ==========================

def classify_metadata(metadata):
    corrupts = []
    lowres = []
    hashes = {}
    duplicates = []

    for path, cls, ok, dims, h in metadata:

        if not ok:
            corrupts.append((path, cls))
            continue

        if dims[0] < MIN_DIM or dims[1] < MIN_DIM:
            lowres.append((path, cls))
            continue

        if h in hashes:
            duplicates.append((path, cls))
        else:
            hashes[h] = (path, cls)

    return corrupts, lowres, duplicates, hashes


# ==========================
# PROCESSAMENTO (RGB + GRAY)
# ==========================

def load_and_clean(path):
    b = Path(path).read_bytes()
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_AREA)
    return img


def process_image(args):
    path, cls = args

    img = load_and_clean(path)
    if img is None:
        return None

    # salva RGB melhorado
    out_ok_cls = DIR_OK / cls
    out_ok_cls.mkdir(exist_ok=True)
    rgb_path = out_ok_cls / (Path(path).stem + ".jpg")
    cv2.imwrite(str(rgb_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # salva cinza
    out_gray_cls = DIR_GRAY / cls
    out_gray_cls.mkdir(exist_ok=True)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = out_gray_cls / (Path(path).stem + ".jpg")
    cv2.imwrite(str(gray_path), gray_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    return str(rgb_path)


# ==========================
# PIPELINE
# ==========================

def run_pipeline():

    print("\n📌 Listando arquivos...")
    files = list_all_image_files(BASE_DIR)

    print(f"Total encontrado: {len(files)}")

    meta = run_inspection(files)
    corrupts, lowres, duplicates, hashes = classify_metadata(meta)

    print(f"\nCorrompidas: {len(corrupts)}")
    print(f"Baixa resolução: {len(lowres)}")
    print(f"Duplicadas: {len(duplicates)}")
    print(f"Válidas únicas: {len(hashes)}")

    # mover corrompidas
    for p, cls in corrupts:
        shutil.copy(p, DIR_CORR / (cls + "_" + Path(p).name))

    # mover baixa resoluçao
    for p, cls in lowres:
        shutil.copy(p, DIR_LOW / (cls + "_" + Path(p).name))

    # mover duplicatas
    for p, cls in duplicates:
        shutil.copy(p, DIR_DUP / (cls + "_" + Path(p).name))

    # processar só válidas
    valid = [(p, cls) for (p, cls) in hashes.values()]

    print(f"\nProcessando imagens válidas: {len(valid)}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = [exe.submit(process_image, v) for v in valid]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("\n✅ PIPELINE FINALIZADO!")
    print(f"Resultados em: {OUT_BASE}")


# ==========================
# EXECUÇÃO
# ==========================

if __name__ == "__main__":
    run_pipeline()
