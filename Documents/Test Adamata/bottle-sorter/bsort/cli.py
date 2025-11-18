# bsort/cli.py

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# Import modul internal
from bsort.data_utils import relabel_and_split_dataset
from bsort.model import run_training, run_inference

def load_config(config_path: str) -> Dict[str, Any]:
    """
    [Google Style Docstring] Memuat konfigurasi dari file YAML.

    Args:
        config_path: Path ke file settings.yaml.

    Returns:
        Kamus (Dict) berisi parameter konfigurasi.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"CLI: Error: File konfigurasi tidak ditemukan di {config_path}")
        sys.exit(1)

def cli_main():
    """
    Program utama CLI yang dapat dipanggil sebagai 'bsort'.
    """
    parser = argparse.ArgumentParser(description="bsort: Real-time Bottle Cap Sorter ML Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sub-parser untuk 'train' (melakukan setup dan training)
    train_parser = subparsers.add_parser("train", help="Menyiapkan data (relabeling) dan melatih model deteksi objek.")
    train_parser.add_argument("--config", type=str, default="settings.yaml", help="Path ke file settings.yaml.")

    # Sub-parser untuk 'infer'
    infer_parser = subparsers.add_parser("infer", help="Melakukan inferensi pada gambar input.")
    infer_parser.add_argument("--config", type=str, default="settings.yaml", help="Path ke file settings.yaml.")
    infer_parser.add_argument("--image", type=str, required=True, help="Path ke gambar input untuk inferensi.")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "train":
        print("--- CLI: Memulai proses SETUP dan TRAINING ---")
        relabel_and_split_dataset(config)
        run_training(config)

    elif args.command == "infer":
        print("--- CLI: Memulai proses INFERENSI ---")
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"CLI: Error: File gambar tidak ditemukan di {image_path}")
            sys.exit(1)
            
        run_inference(config, image_path)

if __name__ == "__main__":
    cli_main()