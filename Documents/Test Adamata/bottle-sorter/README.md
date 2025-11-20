Bottle Cap Sorter ML Pipeline

Proyek ini mengimplementasikan pipeline Machine Learning (ML) end-to-end untuk mendeteksi tutup botol berdasarkan warna, dengan fokus pada pemenuhan kendala inferensi real-time 5-10 ms pada perangkat edge (Raspberry Pi 5).

Analisis Model & Hasil Eksperimen:
A. Diagnosa Overfitting (Isu Data Kritis)Model dipilih dari dua eksperimen fine-tuning (Baseline vs. Revisi) yang dijalankan pada data yang sangat terbatas (Jumlah File=12).
- Diagnosis: Hasil mAP yang sangat tinggi (0.9950) adalah bukti dari overfitting yang ekstrem. Model menghafal data validation karena jumlah sampel yang tidak memadai, bukan karena kemampuan generalisasi yang baik.
- Mitigasi: Eksperimen Revisi mencoba menggunakan regularization yang lebih kuat (weight decay tinggi) untuk mengatasi overfitting, namun perbedaan hasilnya diabaikan, mengonfirmasi data starvation sebagai masalah utama.

Keputusan Model Deployment:
Karena akurasi (mAP) kedua model hampir identik, keputusan didasarkan pada faktor yang paling kritis untuk edge deployment: Kecepatan.
Model Baseline memiliki speed 73.5303 ms, dibandingkan dengan Model Revisi 78.3922. Artinya model baseline lebih cepat sehingga model ini yang digunakan.

Strategi Pemenuhan Kendala Real-Time:
Waktu inferensi yang diukur pada host CPU Anda adalah 73.5303 ms, yang jauh dari target 5-10ms. Solusi terletak pada Optimasi Pasca-Training.

Model yang Digunakan: YOLOv8n (nano) Solusi Teknis Wajib:
1. Quantization (INT8): Model terpilih harus di-quantize dari presisi FP32 menjadi INT8. Ini adalah teknik paling efektif untuk meningkatkan kecepatan inference pada chip Raspberry Pi 5 yang dioptimalkan untuk komputasi integer.
2. Runtime Akselerasi: Model harus diekspor ke format ONNX atau TFLite dan dijalankan menggunakan runtime akselerasi (seperti ONNX Runtime atau OpenVINO) di RPi 5.
   
Kombinasi kedua langkah ini akan memberikan peningkatan kecepatan total 6x hingga 10x, membawa performa model ke dalam ambang batas kendala 5-10 ms.

Deployment Infrastructure:
Proyek ini telah dikonfigurasi untuk deployment melalui container Docker dan Continuous Integration.

A. Containerization (Dockerfile)
Image Docker dibuat untuk mengemas aplikasi (bsort/) dan weights model terpilih (best.pt) dalam lingkungan runtime yang stabil dan portabel (berbasis Python 3.10-slim).

B. CI/CD Pipeline (GitHub Actions)
Workflow CI/CD diimplementasikan untuk memastikan kualitas kode dan deployment yang andal:
- Code Quality: Menjalankan linting (Pylint) dan formatting (Black, isort) pada setiap push.
- Build Verification: Memastikan image Docker dapat dibangun dengan sukses di server GitHub Actions.

WandB Model Tracking URL:
Saya mengalami masalah dengan visibilitas server WandB yang menyebabkan kesalahan 404 di halaman profil pribadi saya. Namun, semua data pelatihan berhasil dicatat dan seharusnya dapat diakses melalui tautan proyek yang disediakan. Analisis lengkapnya disajikan dalam Jupyter Notebook yang telah dikirimkan.
