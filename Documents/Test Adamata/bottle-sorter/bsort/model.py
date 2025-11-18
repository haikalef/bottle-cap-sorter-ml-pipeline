# bsort/model.py

import torch
import time
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_training(config: Dict[str, Any]) -> Path:
    """
    [Google Style Docstring] Melakukan fine-tuning model YOLOv8n berdasarkan konfigurasi
    (menggunakan parameter regularisasi yang ditingkatkan).

    Args:
        config: Kamus konfigurasi dari settings.yaml.

    Returns:
        Path ke weights model terbaik (best.pt).
    """
    print("\n--- Model: Mulai Training Model YOLOv8n ---")
    
    train_config = config['training']
    paths_config = config['paths']
    
    # seed untuk konsistensi training
    YOLO.seed = 42
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=paths_config['data_config_file'],    
        epochs=train_config['epochs'],              
        imgsz=paths_config['img_size'],
        batch=train_config['batch_size'],
        lr0=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        project=paths_config['project_dir'].replace('./', ''),
        name='yolov8n_deploy_run',
        exist_ok=True
    )
    
    final_model_path = results.save_dir / 'weights' / 'best.pt'
    print(f"Model: Training Selesai. Model terbaik disimpan di: {final_model_path}")
    return final_model_path

def run_inference(config: Dict[str, Any], image_path: Path) -> None:
    """
    [Google Style Docstring] Melakukan inferensi deteksi objek pada satu gambar 
    dan menyimpan hasilnya.

    Args:
        config: Kamus konfigurasi dari settings.yaml.
        image_path: Path ke file gambar input.
    """
    print(f"\n--- Model: Mulai Inferensi pada Gambar: {image_path.name} ---")
    
    model_deploy_path = config['inference']['model_path']
    
    try:
        model = YOLO(model_deploy_path)
        model.to(DEVICE)
    except Exception as e:
        print(f"Model: Gagal memuat model dari {model_deploy_path}. Error: {e}")
        return

    # Lakukan prediksi dan simpan hasilnya
    model.predict(
        source=str(image_path),
        imgsz=config['paths']['img_size'],
        conf=config['inference']['confidence_threshold'],
        device=DEVICE,
        save=True,
        project=config['inference']['output_dir'],
        name='results',
        exist_ok=True
    )
    
    print(f"Model: Inferensi Selesai. Hasil disimpan di: {config['inference']['output_dir']}/results")