!pip install kaggle tqdm --quiet

from google.colab import drive
drive.mount('/content/drive')

! mkdir ~/.kaggle

! cp /content/drive/MyDrive/kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download paultimothymooney/blood-cells

! unzip blood-cells.zip

# Pipeline: ETAPA 2 (Coleta & Organização) + ETAPA 3 (Pré-processamento + Features)
# Colar no Colab; ajuste apenas BASE_DIR e OUT_BASE se desejar.

import os
from pathlib import Path
import shutil
import hashlib
import random
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from skimage.feature import local_binary_pattern

# ----------------------
# CONFIGURAÇÃO
# ----------------------
BASE_DIR = Path("/content/dataset2-master/dataset2-master/images/TEST")  # onde estão as classes originais
OUT_BASE = Path("/content/dataset_processed")                           # onde tudo será salvo
OUT_BASE.mkdir(parents=True, exist_ok=True)

# Pastas de resultado (Etapa 2)
DIR_CORR = OUT_BASE / "CORROMPIDAS"
DIR_LOW  = OUT_BASE / "BAIXA_RES"
DIR_DUP  = OUT_BASE / "DUPLICADAS"
RAW_CLEANED = OUT_BASE / "RAW_CLEANED"   # conterá train/ e test/ com subpastas por classe
FEATURE_DIR = OUT_BASE / "features"

for d in [DIR_CORR, DIR_LOW, DIR_DUP, RAW_CLEANED, FEATURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# parâmetros
MIN_DIM = 64
TRAIN_RATIO = 0.8
IMAGE_SIZE = (256, 256)
JPEG_QUALITY = 95

# ----------------------
# UTILITÁRIOS (inspecionar arquivos)
# ----------------------
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

def inspect_file(tuple_cls_path):
    """Retorna (path_str, cls, ok_bool, (w,h) or None, hash or None)"""
    cls, path = tuple_cls_path
    p = Path(path)
    try:
        b = p.read_bytes()
        if len(b) == 0:
            return (str(p), cls, False, None, None)
        h = sha256_of_bytes(b)
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return (str(p), cls, False, None, h)
        hgt, wid = img.shape[0], img.shape[1]
        return (str(p), cls, True, (wid, hgt), h)
    except Exception:
        return (str(p), cls, False, None, None)

# ----------------------
# ETAPA 2: organizar, remover corruptas/lowres/duplicatas, anonymize (regravando)
# ----------------------
def etapa2_organize_and_clean():
    print("Etapa 2 — Inspeção e organização dos arquivos...")
    files = list_all_image_files(BASE_DIR)
    print(f"Arquivos encontrados: {len(files)}")

    metadata = []
    for t in tqdm(files, desc="Inspecionando"):
        metadata.append(inspect_file(t))

    # classificar
    corrupts = []
    lowres = []
    duplicates = []
    hashes = {}
    valid_unique = []

    for path, cls, ok, dims, h in metadata:
        if not ok:
            corrupts.append((path, cls))
            continue
        if dims is None or dims[0] < MIN_DIM or dims[1] < MIN_DIM:
            lowres.append((path, cls))
            continue
        if h is None:
            corrupts.append((path, cls))
            continue
        if h in hashes:
            duplicates.append((path, cls))
        else:
            hashes[h] = (path, cls)
            valid_unique.append((path, cls))

    # mover arquivos problemáticos (copiar para registro; mantém originais)
    for p, cls in corrupts:
        shutil.copy(p, DIR_CORR / (cls + "_" + Path(p).name))
    for p, cls in lowres:
        shutil.copy(p, DIR_LOW / (cls + "_" + Path(p).name))
    for p, cls in duplicates:
        shutil.copy(p, DIR_DUP / (cls + "_" + Path(p).name))

    print(f"Corrompidos: {len(corrupts)} | Baixa-res: {len(lowres)} | Duplicadas: {len(duplicates)}")
    print(f"Válidas únicas: {len(valid_unique)}")

    # Regravar imagens válidas sem metadados e organizar em RAW_CLEANED/<train|test>/<class>/
    # primeiro agrupar por classe
    by_class = {}
    for p, cls in valid_unique:
        by_class.setdefault(cls, []).append(p)

    for cls, paths in by_class.items():
        random.shuffle(paths)
        n_train = int(len(paths) * TRAIN_RATIO)
        train_paths = paths[:n_train]
        test_paths = paths[n_train:]

        for kind, subset in (("train", train_paths), ("test", test_paths)):
            dest_root = RAW_CLEANED / kind / cls
            dest_root.mkdir(parents=True, exist_ok=True)
            for src in subset:
                src_p = Path(src)
                # reabrir e regravar para remover EXIF/metadados e padronizar formato (JPEG)
                img = cv2.imread(str(src_p))
                if img is None:
                    continue
                # opcional: resize aqui para reduzir tamanho do RAW_CLEANED (mantemos original dims)
                out_path = dest_root / (src_p.stem + ".jpg")
                cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    print("Etapa 2 concluída. RAW_CLEANED com train/test criado em:", RAW_CLEANED)
    return {
        "corrupts": corrupts,
        "lowres": lowres,
        "duplicates": duplicates,
        "valid_counts": {k: len(v) for k, v in by_class.items()}
    }

# ----------------------
# ETAPA 3: pré-processamento, augmentation, extração de features e salvamento
# ----------------------

# Manual GLCM (compatível com scikit-image 0.25.x)
def manual_glcm_features(image_gray, levels=8, distances=[1], angles=[0]):
    """
    image_gray: uint8 gray image (0..255)
    levels: quantization levels (e.g., 8)
    returns: dict with contrast, dissimilarity, homogeneity, energy, ASM, correlation
    """
    # quantize
    q = (image_gray.astype(np.float32) / 256.0 * (levels-1)).astype(np.uint8)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = q.shape
    for d in distances:
        for angle in angles:
            dx = int(round(np.cos(angle) * d))
            dy = int(round(-np.sin(angle) * d))  # careful y direction
            for y in range(h):
                for x in range(w):
                    x2 = x + dx
                    y2 = y + dy
                    if 0 <= x2 < w and 0 <= y2 < h:
                        i = q[y, x]
                        j = q[y2, x2]
                        glcm[i, j] += 1
    if glcm.sum() == 0:
        glcm = glcm + 1e-10
    glcm = glcm / glcm.sum()

    i_idx = np.arange(levels)
    j_idx = np.arange(levels)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')

    contrast = np.sum(glcm * (I - J)**2)
    dissimilarity = np.sum(glcm * np.abs(I - J))
    homogeneity = np.sum(glcm / (1.0 + (I - J)**2))
    asm = np.sum(glcm**2)
    energy = np.sqrt(asm)
    # correlation
    mean_i = np.sum(I * glcm)
    mean_j = np.sum(J * glcm)
    std_i = np.sqrt(np.sum(((I - mean_i)**2) * glcm))
    std_j = np.sqrt(np.sum(((J - mean_j)**2) * glcm))
    if std_i * std_j == 0:
        correlation = 0.0
    else:
        correlation = np.sum((I - mean_i) * (J - mean_j) * glcm) / (std_i * std_j)
    return {
        "glcm_contrast": float(contrast),
        "glcm_dissimilarity": float(dissimilarity),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_asm": float(asm),
        "glcm_correlation": float(correlation)
    }

# LBP features
def lbp_hist_features(gray, P=8, R=1, n_bins=None):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    if n_bins is None:
        n_bins = int(P + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float32")
    if hist.sum() > 0:
        hist /= hist.sum()
    return {f"lbp_{i}": float(hist[i]) for i in range(len(hist))}

# color statistics
def color_stats(img_bgr):
    mean = img_bgr.mean(axis=(0,1))
    var = img_bgr.var(axis=(0,1))
    return {
        "R_mean": float(mean[2]), "G_mean": float(mean[1]), "B_mean": float(mean[0]),
        "R_var": float(var[2]), "G_var": float(var[1]), "B_var": float(var[0])
    }

# shape metrics: area, perimeter, circularity (largest contour)
def shape_metrics(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0.0, "perimeter": 0.0, "circularity": 0.0}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True)) if cv2.arcLength(c, True) > 0 else 1.0
    circ = float((4.0 * np.pi * area) / (per * per + 1e-8))
    return {"area": area, "perimeter": per, "circularity": circ}

# histogram features (intensity distribution)
def hist_features(gray, nbins=32):
    hist = cv2.calcHist([gray],[0],None,[nbins],[0,256]).flatten().astype("float32")
    if hist.sum() > 0:
        hist /= hist.sum()
    return {f"hist_{i}": float(hist[i]) for i in range(nbins)}

# augmentations to apply (on the cleaned/resized image)
def generate_augmentations(img):
    # expects BGR uint8 same size
    a = {}
    a["orig"] = img
    a["flip_h"] = cv2.flip(img, 1)
    a["flip_v"] = cv2.flip(img, 0)
    a["rot_15"] = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 15, 1.0), (img.shape[1], img.shape[0]))
    a["rot_-15"] = cv2.warpAffine(img, cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), -15, 1.0), (img.shape[1], img.shape[0]))
    a["bright_up"] = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    a["bright_down"] = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    return a

# main extraction for one image (returns list of rows for CSV)
def extract_features_for_image(img_bgr, filename, cls):
    rows = []
    # preprocess: resize, normalize intensity, denoise
    img_resized = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    # intensity normalization (0-255)
    img_resized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_denoise = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)

    aug = generate_augmentations(img_denoise)
    for aug_name, aug_img in aug.items():
        gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
        feat = {
            "file": filename,
            "class": cls,
            "augment": aug_name
        }
        feat.update(color_stats(aug_img))
        feat.update(lbp_hist_features(gray, P=8, R=1, n_bins=10))
        feat.update(manual_glcm_features(gray, levels=8, distances=[1], angles=[0]))
        feat.update(shape_metrics(gray))
        feat.update(hist_features(gray, nbins=32))
        rows.append(feat)
    return rows

# ----------------------
# Run Etapa 3 over RAW_CLEANED (train + test)
# ----------------------
def etapa3_extract_all_features():
    print("Etapa 3 — extraindo features a partir de RAW_CLEANED (train/test)...")
    all_rows = []
    # walk train and test subfolders
    for split in ("train", "test"):
        split_root = RAW_CLEANED / split
        if not split_root.exists():
            continue
        for cls_folder in sorted([p for p in split_root.iterdir() if p.is_dir()]):
            cls = cls_folder.name
            for img_path in cls_folder.glob("*"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rows = extract_features_for_image(img, img_path.name, cls)
                all_rows.extend(rows)

    if not all_rows:
        print("Nenhuma feature extraída.")
        return

    df = pd.DataFrame(all_rows)
    # salvar CSV
    csv_path = FEATURE_DIR / "features.csv"
    df.to_csv(csv_path, index=False)

    # salvar NPY (apenas matrizes float32 de features) + labels separadas
    feat_cols = [c for c in df.columns if c not in ("file","class","augment")]
    features_array = df[feat_cols].to_numpy(dtype=np.float32)
    labels_array = df[["file","class","augment"]].to_numpy()
    np.save(str(FEATURE_DIR / "features.npy"), features_array)
    np.save(str(FEATURE_DIR / "labels.npy"), labels_array)

    print("Features salvas em:", csv_path, "e NPY em:", FEATURE_DIR / "features.npy")
    return df

# ----------------------
# MAIN: executar Etapa2 + Etapa3
# ----------------------
if __name__ == "__main__":
    # 1) Etapa 2: inspeção, limpeza e organização com train/test
    summary = etapa2_organize_and_clean()

    # 2) Etapa 3: pré-processamento, augmentations, extração de features
    df_features = etapa3_extract_all_features()

    print("Pipeline completo. Resultado em:", OUT_BASE)
