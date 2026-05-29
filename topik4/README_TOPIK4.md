# Topik 4 Bonus - Eksperimen AI/Machine Learning di GPU RTX 3080

## Tujuan

Topik ini meminta survey algoritma AI/Machine Learning yang dapat berjalan di GPU, investigasi tools/framework NVIDIA yang dapat memanfaatkan RTX 3080, lalu melakukan uji coba pada GPU03. Paket ini berisi eksperimen siap jalan untuk membandingkan training model AI di CPU dan GPU.

## Masalah yang dipilih

Masalah yang dipilih adalah **klasifikasi tabular sintetis multi-kelas** menggunakan model **Multi-Layer Perceptron (MLP)** di PyTorch.

Alasan pemilihan:

1. Tidak membutuhkan dataset eksternal atau internet.
2. Bisa dijalankan ulang dengan seed tetap.
3. Cocok untuk GPU karena training MLP didominasi operasi matriks/tensor.
4. Output mudah dipahami untuk laporan: waktu training, akurasi, speedup, dan memori GPU.
5. RTX 3080 mendukung Tensor Cores, sehingga mode AMP/mixed precision relevan untuk diuji.

## File

- `topik4_gpu_ai_experiment.py`  
  Program utama eksperimen.

- `run_topik4_experiment.sh`  
  Script untuk menjalankan eksperimen dan menyimpan output terminal ke file teks.

- `README_TOPIK4.md`  
  Penjelasan eksperimen dan cara pakai.

## Cara menjalankan di GPU03

```bash
unzip topik4_gpu_ai_bonus.zip
cd topik4_gpu_ai_bonus
bash run_topik4_experiment.sh
```

Hasil akan tersimpan sebagai:

```text
topik4_output_YYYYMMDD_HHMMSS.txt
topik4_results.json
```

File `.txt` bisa langsung diberikan ke AI sebagai context untuk membantu membuat laporan.

## Survey singkat tools/framework AI-ML GPU NVIDIA

| Tool/Framework | Kegunaan | Backend GPU yang umum dipakai |
|---|---|---|
| PyTorch | Training dan inference deep learning: MLP, CNN, RNN, Transformer | CUDA, cuDNN, cuBLAS, AMP |
| TensorFlow/Keras | Training dan inference deep learning | CUDA, cuDNN, XLA |
| TensorRT | Optimasi inference model terlatih | CUDA, FP32/FP16/INT8 engine optimization |
| RAPIDS cuML | Machine learning klasik: Random Forest, KMeans, PCA, UMAP, beberapa SVM | CUDA-X Data Science |
| CuPy | Komputasi array seperti NumPy tetapi di GPU | CUDA kernels, cuBLAS, cuFFT |
| Numba CUDA | Membuat custom CUDA kernel dari Python | CUDA programming |
| NVIDIA DALI | Pipeline data loading/preprocessing untuk deep learning | CUDA acceleration |

## Catatan environment RTX 3080

RTX 3080 adalah GPU NVIDIA generasi Ampere. Untuk PyTorch, TensorFlow, atau RAPIDS, environment harus memakai driver dan CUDA runtime yang mendukung GPU tersebut. Jika PyTorch tidak mendeteksi CUDA, program tetap menjalankan CPU baseline dan memberi pesan diagnosis.

Cek GPU:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
PY
```

## Output yang dihasilkan program

Program mencetak:

1. Survey singkat framework AI/ML GPU.
2. Informasi environment: versi Python, PyTorch, CUDA, cuDNN, nama GPU, compute capability, dan memori GPU.
3. Detail dataset sintetis.
4. Hasil training CPU FP32.
5. Hasil training GPU CUDA FP32, jika CUDA tersedia.
6. Hasil training GPU CUDA AMP/mixed precision, jika CUDA tersedia.
7. Ringkasan tabel: waktu training, rata-rata waktu epoch, akurasi test, memori GPU.
8. Speedup GPU terhadap CPU.
9. Kesimpulan otomatis.

## Contoh interpretasi untuk laporan

- Jika GPU FP32 lebih cepat dari CPU, berarti operasi training neural network berhasil diparalelkan dengan CUDA.
- Jika GPU AMP lebih cepat dari GPU FP32, berarti mixed precision memberi keuntungan, biasanya karena Tensor Cores pada RTX 3080 dapat dimanfaatkan.
- Jika akurasi CPU dan GPU mirip, berarti hasil training konsisten walaupun detail operasi floating-point berbeda.
- Jika GPU tidak lebih cepat, kemungkinan ukuran model/data terlalu kecil, overhead transfer data masih dominan, atau environment CUDA/PyTorch belum optimal.

## Modifikasi ukuran eksperimen

Kalau runtime terlalu lama:

```bash
python3 topik4_gpu_ai_experiment.py --samples 50000 --epochs 5
```

Kalau ingin workload lebih berat:

```bash
python3 topik4_gpu_ai_experiment.py --samples 500000 --features 256 --epochs 10 --batch-size 4096
```
