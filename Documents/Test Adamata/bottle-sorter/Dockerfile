# Dockerfile
# Menggunakan Python base image yang ringan
FROM python:3.10-slim

# Menetapkan direktori kerja
WORKDIR /app

# Menyalin requirements.txt dan menginstal dependensi
COPY requirements.txt .
# Instalasi library sistem yang dibutuhkan oleh OpenCV
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

# Menyalin file kode, config, dan weights
COPY settings.yaml .
COPY data_config.yaml .
COPY bsort /app/bsort/

# Menyalin weights model terbaik (Baseline) ke dalam container
COPY ./ada-mata-bottle-cap-sorting/yolov8n_baseline_run/weights/best.pt /app/best.pt

# Mendefinisikan entry point untuk skrip CLI
ENTRYPOINT ["python", "-m", "bsort.cli"]

# Perintah default untuk menjalankan inferensi dengan path relatif
# Ganti nama file test jika diperlukan
CMD ["infer", "--config", "settings.yaml", "--image", "data/bottles_relabelled/images/test/raw-250110_dc_s001_b2_15.jpg"]