#!/usr/bin/env python3
"""
Topik 4 Bonus - Eksperimen AI/Machine Learning di GPU RTX 3080
Masalah: klasifikasi tabular sintetis menggunakan Multi-Layer Perceptron (MLP) PyTorch.

Output program dibuat naratif agar mudah dipakai sebagai context laporan.
Program membandingkan:
1) CPU FP32
2) GPU CUDA FP32, jika tersedia
3) GPU CUDA AMP/mixed precision, jika tersedia

Tidak membutuhkan internet dan tidak membutuhkan dataset eksternal.
"""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    print("ERROR: PyTorch tidak bisa di-import.")
    print("Detail:", repr(exc))
    print("Install PyTorch dengan dukungan CUDA yang sesuai untuk RTX 3080, lalu jalankan ulang.")
    sys.exit(1)


@dataclass
class RunResult:
    name: str
    device: str
    precision: str
    epochs: int
    train_seconds: float
    avg_epoch_seconds: float
    test_accuracy: float
    final_loss: float
    max_gpu_memory_mb: Optional[float]
    notes: str


class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception as exc:
        return f"Tidak tersedia ({exc})"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(n_samples: int, n_features: int, n_classes: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dataset sintetis non-linear untuk klasifikasi multi-kelas."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    x = torch.randn(n_samples, n_features, generator=gen)
    w1 = torch.randn(n_features, n_classes, generator=gen)
    w2 = torch.randn(n_features, n_classes, generator=gen)

    # Kombinasi linear + non-linear agar MLP relevan, bukan sekadar linear classifier.
    logits = x @ w1 + 0.35 * torch.sin(x @ w2) + 0.10 * torch.randn(n_samples, n_classes, generator=gen)
    y = torch.argmax(logits, dim=1).long()
    return x.float(), y


def split_dataset(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.8):
    n_train = int(x.shape[0] * train_ratio)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device, use_amp: bool) -> float:
    model.eval()
    correct = 0
    total = 0
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device, non_blocking=True)
            yb = y[start:start + batch_size].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def train_one_setting(
    name: str,
    device_str: str,
    precision: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    args: argparse.Namespace,
) -> RunResult:
    device = torch.device(device_str)
    use_amp = precision.lower() == "amp"
    amp_enabled = use_amp and device.type == "cuda"

    set_seed(args.seed)
    model = MLP(args.features, args.classes, args.hidden1, args.hidden2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    n = x_train.shape[0]
    final_loss = math.nan
    epoch_times: List[float] = []

    start_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        start_epoch = time.perf_counter()

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(xb)
                loss = criterion(out, yb)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.detach().float().item() * yb.numel()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_seconds = time.perf_counter() - start_epoch
        epoch_times.append(epoch_seconds)
        final_loss = epoch_loss / n
        print(f"  Epoch {epoch:02d}/{args.epochs} | loss={final_loss:.6f} | waktu_epoch={epoch_seconds:.3f} detik")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start_total

    acc = accuracy(model, x_test, y_test, args.batch_size, device, use_amp)
    max_mem = None
    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    note = "Berhasil"
    if amp_enabled:
        note = "Berhasil dengan Automatic Mixed Precision; pada RTX 3080 ini dapat memakai Tensor Cores."

    return RunResult(
        name=name,
        device=device_str,
        precision=precision,
        epochs=args.epochs,
        train_seconds=train_seconds,
        avg_epoch_seconds=sum(epoch_times) / len(epoch_times),
        test_accuracy=acc,
        final_loss=final_loss,
        max_gpu_memory_mb=max_mem,
        notes=note,
    )


def collect_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": run_cmd(["nvidia-smi"]),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mb": round(props.total_memory / (1024 ** 2), 2),
            "multi_processor_count": props.multi_processor_count,
        })
    return env


def print_survey() -> None:
    print("=" * 78)
    print("SURVEY SINGKAT: ALGORITMA/FRAMEWORK AI-ML YANG BISA BERPROSES DI GPU NVIDIA")
    print("=" * 78)
    rows = [
        ("PyTorch", "Deep learning: MLP, CNN, RNN, Transformer", "CUDA, cuDNN, cuBLAS, AMP"),
        ("TensorFlow/Keras", "Deep learning training dan inference", "CUDA, cuDNN, XLA"),
        ("TensorRT", "Optimasi inference model terlatih", "FP32/FP16/INT8, engine optimized"),
        ("RAPIDS cuML", "ML klasik: Random Forest, KMeans, PCA, UMAP, SVM tertentu", "CUDA-X Data Science"),
        ("CuPy", "Array/Numpy-like computation di GPU", "CUDA kernels, cuBLAS, cuFFT"),
        ("Numba CUDA", "Custom kernel Python untuk GPU", "CUDA programming dari Python"),
        ("scikit-learn-intelex/cuML accel", "Akselerasi pipeline ML tertentu", "Backend GPU/accelerated libraries"),
    ]
    print(f"{'Framework/Tool':<24} | {'Contoh algoritma':<48} | Backend")
    print("-" * 110)
    for a, b, c in rows:
        print(f"{a:<24} | {b:<48} | {c}")
    print()


def print_environment(env: Dict[str, Any]) -> None:
    print("=" * 78)
    print("INFORMASI ENVIRONMENT")
    print("=" * 78)
    keys = [
        "python", "platform", "torch_version", "torch_cuda_version", "cudnn_version",
        "cuda_available", "gpu_name", "gpu_count", "compute_capability", "total_memory_mb",
        "multi_processor_count",
    ]
    for k in keys:
        if k in env:
            print(f"{k:24s}: {env[k]}")
    print()


def print_summary(results: List[RunResult], args: argparse.Namespace) -> None:
    print("=" * 78)
    print("RINGKASAN HASIL EKSPERIMEN")
    print("=" * 78)
    print(f"Masalah             : Klasifikasi tabular sintetis multi-kelas")
    print(f"Model               : MLP ({args.features} -> {args.hidden1} -> {args.hidden2} -> {args.classes})")
    print(f"Jumlah data         : {args.samples:,} sampel, {args.features} fitur, {args.classes} kelas")
    print(f"Epoch               : {args.epochs}")
    print(f"Batch size          : {args.batch_size}")
    print()
    print(f"{'Setting':<20} | {'Precision':<10} | {'Train(s)':>10} | {'Epoch avg(s)':>12} | {'Akurasi':>9} | {'GPU Mem(MB)':>11}")
    print("-" * 86)
    for r in results:
        mem = "-" if r.max_gpu_memory_mb is None else f"{r.max_gpu_memory_mb:.1f}"
        print(f"{r.name:<20} | {r.precision:<10} | {r.train_seconds:10.3f} | {r.avg_epoch_seconds:12.3f} | {r.test_accuracy:9.4f} | {mem:>11}")

    cpu = next((r for r in results if r.device == "cpu"), None)
    if cpu:
        print()
        print("Perbandingan speedup terhadap CPU:")
        for r in results:
            if r is cpu:
                continue
            speedup = cpu.train_seconds / r.train_seconds if r.train_seconds > 0 else float("nan")
            print(f"  - {r.name}: {speedup:.2f}x lebih cepat dari CPU berdasarkan waktu training total")

    print()
    print("Kesimpulan otomatis:")
    if len(results) == 1:
        print("  - Eksperimen hanya berjalan di CPU karena CUDA/GPU tidak terdeteksi oleh PyTorch.")
        print("  - Untuk RTX 3080, pastikan driver NVIDIA aktif dan PyTorch terpasang dengan CUDA yang mendukung compute capability 8.6.")
    else:
        best = min(results, key=lambda r: r.train_seconds)
        print(f"  - Setting tercepat pada eksperimen ini adalah {best.name} ({best.precision}) dengan waktu training {best.train_seconds:.3f} detik.")
        print("  - GPU cocok untuk workload AI/ML yang banyak melakukan operasi matriks dan tensor secara paralel.")
        print("  - Mixed precision/AMP biasanya semakin menguntungkan pada GPU RTX karena dapat memakai Tensor Cores, tetapi hasil aktual tetap bergantung ukuran model, batch size, dan versi library.")
        print("  - Akurasi CPU dan GPU tidak harus identik 100% karena urutan operasi floating-point berbeda, tetapi harus berada pada kisaran yang sebanding.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksperimen AI/ML GPU untuk Topik 4 Bonus")
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="topik4_results.json")
    args = parser.parse_args()

    print_survey()
    env = collect_environment()
    print_environment(env)

    print("=" * 78)
    print("MEMBUAT DATASET SINTETIS")
    print("=" * 78)
    print("Dataset dibuat offline menggunakan torch.randn, sehingga eksperimen bisa diulang tanpa internet.")
    print("Label dibuat dari fungsi linear + non-linear, sehingga model MLP punya masalah klasifikasi yang realistis untuk diuji.")
    x, y = make_dataset(args.samples, args.features, args.classes, args.seed)
    x_train, y_train, x_test, y_test = split_dataset(x, y)
    print(f"Train: {x_train.shape[0]:,} sampel | Test: {x_test.shape[0]:,} sampel")
    print()

    results: List[RunResult] = []

    print("=" * 78)
    print("EKSPERIMEN 1: CPU FP32")
    print("=" * 78)
    results.append(train_one_setting("CPU FP32", "cpu", "fp32", x_train, y_train, x_test, y_test, args))
    print()

    if torch.cuda.is_available():
        print("=" * 78)
        print("EKSPERIMEN 2: GPU CUDA FP32")
        print("=" * 78)
        results.append(train_one_setting("GPU CUDA FP32", "cuda", "fp32", x_train, y_train, x_test, y_test, args))
        print()

        print("=" * 78)
        print("EKSPERIMEN 3: GPU CUDA AMP / MIXED PRECISION")
        print("=" * 78)
        results.append(train_one_setting("GPU CUDA AMP", "cuda", "amp", x_train, y_train, x_test, y_test, args))
        print()
    else:
        print("CUDA tidak tersedia menurut PyTorch, jadi eksperimen GPU dilewati.")
        print()

    print_summary(results, args)

    payload = {
        "environment": env,
        "experiment_config": vars(args),
        "results": [asdict(r) for r in results],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"File JSON hasil eksperimen disimpan ke: {args.output_json}")
    print("Simpan seluruh output terminal ini sebagai context untuk penulisan laporan.")


if __name__ == "__main__":
    main()
