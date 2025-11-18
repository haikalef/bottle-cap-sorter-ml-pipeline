import os
import cv2
import numpy as np
import yaml
import sys
import torch
import random
import time
from pathlib import Path
from typing import Dict, Any

# Konstanta Warna HSV
blue_h_min = 90
blue_h_max = 140
light_v_threshold = 160

SEED = 42 

def get_yolo_class(img_hsv: np.ndarray) -> int:
   def get_yolo_class(img_hsv: np.ndarray) -> int:
    """
    Menentukan kelas warna tutup botol (0: Light Blue, 1: Dark Blue, 2: Others) 
    berdasarkan nilai rata-rata HSV dari patch gambar.

    Args:
        img_hsv: Patch gambar tutup botol dalam ruang warna HSV.

    Returns:
        ID kelas baru (0, 1, atau 2).
    """
    if img_hsv.size == 0: return 2
    h_mean, s_mean, v_mean, _ = cv2.mean(img_hsv)
    if blue_h_min <= h_mean <= blue_h_max:
        if v_mean > light_v_threshold: return 0  
        else: return 1  
    else: return 2

def relabel_and_split_dataset(config: Dict[str, Any]) -> None:
    """
    Memproses dataset YOLO asli, menyesuaikan label berdasarkan warna, 
    dan membagi dataset menjadi Train/Val/Test (80:10:10).

    Args:
        config: Kamus konfigurasi dari settings.yaml.
    """
    print("--- DataUtils: Memulai Penyesuaian Label Dataset ---")
    source_path = Path(config['paths']['data_raw_dir'])
    target_path = Path(config['paths']['data_relabelled_dir'])
    train_ratio, val_ratio = 0.8, 0.1

    # 1. Buat folder output
    for split in ['train', 'val', 'test']:
        (target_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    image_files = list((source_path / "images").glob("*.jpg")) 
    
    # 2. Bagi file menggunakan seed untuk konsistensi
    random.seed(SEED)
    random.shuffle(image_files) 
    
    num_files = len(image_files)
    if num_files == 0:
        print(f"Error: Tidak ada file gambar ditemukan di {source_path / 'images'}")
        sys.exit(1)
        
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]
    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    # 3. Proses relabeling
    for split_name, files in splits.items():
        for img_path in files:
            label_name = img_path.stem + '.txt'
            label_path_orig = source_path / "labels" / label_name
            img_path_out = target_path / "images" / split_name / img_path.name
            label_path_out = target_path / "labels" / split_name / label_name
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            img_h, img_w, _ = img.shape
            new_annotations = []
            
            if label_path_orig.exists():
                with open(label_path_orig, 'r') as f: lines = f.readlines()
                for line in lines:
                    try: _, x_c, y_c, w, h = map(float, line.strip().split()) 
                    except ValueError: continue
                    
                    x1 = max(0, int(x_c * img_w - w * img_w / 2))
                    y1 = max(0, int(y_c * img_h - h * img_h / 2))
                    x2 = min(img_w, int(x_c * img_w + w * img_w / 2))
                    y2 = min(img_h, int(y_c * img_h + h * img_h / 2))
                    
                    cap_patch = img[y1:y2, x1:x2]
                    
                    new_class_id = get_yolo_class(cv2.cvtColor(cap_patch, cv2.COLOR_BGR2HSV)) if cap_patch.size > 0 else 2
                    new_annotations.append(f"{new_class_id} {x_c} {y_c} {w} {h}\n")
            
                with open(label_path_out, 'w') as f: f.writelines(new_annotations)
            cv2.imwrite(str(img_path_out), img)

    print(f"DataUtils: Penyesuaian label selesai. Data baru di: {target_path.name}")