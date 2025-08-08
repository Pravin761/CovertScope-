from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class DetectionResult:
    encoding: str
    bitrate_hz: float
    sample_rate_hz: float
    noise_std: float
    dominant_hz: float
    snr_db: float
    peaks_hz: List[float]
    decoded_bits: Optional[str]
    errors: Optional[int]
    compared_bits: Optional[int]
    ber: Optional[float]
    accuracy_pct: Optional[float]


# =========================
# Signal generation/decoding
# =========================
def generate_random_bits(bit_length: int) -> str:
    return "".join(random.choice("01") for _ in range(bit_length))


def sanitize_bits(bits: str) -> str:
    return "".join(ch for ch in bits if ch in "01")


def generate_signal(bits: str, bitrate_hz: float, sample_rate_hz: float, encoding: str, noise_std: float = 0.05) -> np.ndarray:
    bits = sanitize_bits(bits)
    samples_per_bit = max(2, int(round(sample_rate_hz / bitrate_hz)))
    total_samples = samples_per_bit * len(bits)
    signal = np.zeros(total_samples, dtype=np.float64)

    enc = encoding.lower()
    if enc == "ook":
        for i, b in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            signal[start:end] = 1.0 if b == "1" else 0.0

    elif enc == "manchester":
        half = samples_per_bit // 2
        if half == 0:
            raise ValueError("Increase sample_rate or decrease bitrate for Manchester.")
        for i, b in enumerate(bits):
            start = i * samples_per_bit
            mid = start + half
            end = start + samples_per_bit
            if b == "1":
                signal[start:mid] = 1.0
                signal[mid:end] = 0.0
            else:
                signal[start:mid] = 0.0
                signal[mid:end] = 1.0

    elif enc == "pwm":
        duty_one = 0.7
        duty_zero = 0.3
        for i, b in enumerate(bits):
            start = i * samples_per_bit
            end = start + samples_per_bit
            duty = duty_one if b == "1" else duty_zero
            high = start + int(round(duty * samples_per_bit))
            signal[start:high] = 1.0
            signal[high:end] = 0.0
    else:
        raise ValueError("encoding must be one of: OOK, Manchester, PWM")

    if noise_std > 0:
        signal = signal + np.random.normal(0.0, noise_std, size=signal.shape)
    signal = np.clip(signal, 0.0, 1.0)
    return signal


def detect_periodicity(signal: np.ndarray, sample_rate_hz: float) -> Tuple[float, float, List[float], Tuple[np.ndarray, np.ndarray]]:
    f, pxx = periodogram(signal, fs=sample_rate_hz, scaling="density")
    if len(f) < 3:
        return 0.0, 0.0, [], (f, pxx)
    # Exclude DC
    f1, p1 = f[1:], pxx[1:]
    peak_idx = int(np.argmax(p1))
    dominant_hz = float(f1[peak_idx])
    # SNR: peak vs median of rest
    if len(p1) > 10:
        mask = np.ones_like(p1, dtype=bool)
        mask[peak_idx] = False
        noise_floor = float(np.median(p1[mask]))
    else:
        noise_floor = float(np.median(p1))
    noise_floor = max(noise_floor, 1e-12)
    snr_db = 10.0 * np.log10(float(p1[peak_idx]) / noise_floor)

    # Collect top peaks (optional)
    peaks, _ = find_peaks(p1, height=noise_floor * 4.0)
    peaks_hz = [float(f1[i]) for i in peaks]
    return dominant_hz, snr_db, peaks_hz, (f, pxx)


def decode_ook(signal: np.ndarray, sample_rate_hz: float, bitrate_hz: float, threshold: float = 0.5) -> str:
    spb = int(round(sample_rate_hz / bitrate_hz))
    if spb < 2:
        raise ValueError("Too few samples per bit for OOK decode.")
    centers = [int(i * spb + spb * 0.5) for i in range(len(signal) // spb)]
    bits = []
    for c in centers:
        if c >= len(signal):
            break
        bits.append("1" if signal[c] >= threshold else "0")
    return "".join(bits)


def decode_pwm(signal: np.ndarray, sample_rate_hz: float, bitrate_hz: float, threshold: float = 0.5) -> str:
    spb = int(round(sample_rate_hz / bitrate_hz))
    if spb < 2:
        raise ValueError("Too few samples per bit for PWM decode.")
    num_bits = len(signal) // spb
    bits = []
    for i in range(num_bits):
        start = i * spb
        end = start + spb
        duty = float(np.mean(signal[start:end] >= threshold))
        bits.append("1" if duty >= 0.5 else "0")
    return "".join(bits)


def decode_manchester(signal: np.ndarray, sample_rate_hz: float, bitrate_hz: float, threshold: float = 0.5) -> str:
    spb = int(round(sample_rate_hz / bitrate_hz))
    half = spb // 2
    if half < 1:
        raise ValueError("Too few samples per half-bit for Manchester decode.")
    num_bits = len(signal) // spb
    bits = []
    for i in range(num_bits):
        start = i * spb
        mid = start + half
        end = start + spb
        a = float(np.mean(signal[start:mid] >= threshold))
        b = float(np.mean(signal[mid:end] >= threshold))
        if a > 0.5 and b < 0.5:
            bits.append("1")
        elif a < 0.5 and b > 0.5:
            bits.append("0")
        else:
            bits.append("?")  # undecided
    return "".join(bits)


def compute_ber(true_bits: str, decoded_bits: str) -> Tuple[int, int, float, float]:
    true_bits = sanitize_bits(true_bits)
    decoded_bits = sanitize_bits(decoded_bits)
    n = min(len(true_bits), len(decoded_bits))
    if n == 0:
        return 0, 0, math.nan, math.nan
    errors = sum(1 for i in range(n) if true_bits[i] != decoded_bits[i])
    ber = errors / n
    accuracy_pct = 100.0 * (1.0 - ber)
    return errors, n, ber, accuracy_pct


# =========================
# Save/Load utilities
# =========================
def save_signal(path: str, signal: np.ndarray, sample_rate_hz: float, metadata: Optional[Dict] = None) -> None:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    metadata = metadata or {}
    metadata["sample_rate_hz"] = sample_rate_hz
    if ext == ".npy":
        np.save(path, signal)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    elif ext == ".npz":
        np.savez(path, signal=signal, sample_rate_hz=sample_rate_hz, metadata=json.dumps(metadata))
    elif ext == ".csv":
        time = np.arange(len(signal)) / sample_rate_hz
        arr = np.column_stack([time, signal])
        np.savetxt(path, arr, delimiter=",", header="time,amplitude", comments="")
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    else:
        raise ValueError("Unsupported extension. Use .npy, .npz, or .csv")


def load_signal(path: str, fallback_sample_rate_hz: Optional[float] = None) -> Tuple[np.ndarray, float, Dict]:
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        signal = np.load(path)
        meta_path = path + ".json"
        sr = fallback_sample_rate_hz
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                sr = float(meta.get("sample_rate_hz", sr if sr is not None else 0.0))
        if sr is None or sr <= 0:
            raise ValueError("Sample rate not found; pass --sample-rate when loading .npy saved without metadata.")
        return signal, sr, meta
    elif ext == ".npz":
        data = np.load(path, allow_pickle=True)
        signal = data["signal"]
        sr = float(data["sample_rate_hz"])
        meta_json = data.get("metadata")
        meta = json.loads(str(meta_json)) if meta_json is not None else {}
        return signal, sr, meta
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        # Expect columns: time, amplitude
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("CSV must have two columns: time, amplitude")
        time = arr[:, 0]
        signal = arr[:, 1]
        # Infer sample rate from time vector if possible
        if len(time) >= 2:
            dt = np.median(np.diff(time))
            sr = 1.0 / dt if dt > 0 else (fallback_sample_rate_hz or 0.0)
        else:
            sr = fallback_sample_rate_hz or 0.0
        meta_path = path + ".json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        if sr <= 0:
            raise ValueError("Sample rate not found; pass --sample-rate when loading CSV without time data.")
        return signal, sr, meta
    else:
        raise ValueError("Unsupported extension. Use .npy, .npz, or .csv")


# =========================
# Plotting
# =========================
def plot_signal(signal: np.ndarray, sample_rate_hz: float, title: str = "Signal") -> None:
    plt.figure(figsize=(12, 3.5))
    time = np.arange(len(signal)) / sample_rate_hz
    plt.plot(time, signal, lw=1.2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()


def plot_decoded_bits(decoded_bits: str, bitrate_hz: float, title: str = "Decoded Bits (step)") -> None:
    if not decoded_bits:
        return
    step = 1.0 / bitrate_hz
    t = [0.0]
    y = [0.0]
    cur = 0.0
    for b in decoded_bits:
        bit_val = 1.0 if b == "1" else 0.0
        y.extend([bit_val, bit_val])
        t.extend([cur, cur + step])
        cur += step
    plt.figure(figsize=(12, 2.5))
    plt.step(t, y[1:] + [y[-1]], where="post")
    plt.ylim(-0.2, 1.2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Bit")
    plt.grid(True)
    plt.tight_layout()


def plot_psd(psd: Tuple[np.ndarray, np.ndarray], title: str = "Power Spectral Density") -> None:
    f, pxx = psd
    plt.figure(figsize=(12, 3.5))
    plt.semilogy(f + 1e-9, pxx + 1e-18)  # avoid log(0)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()


def plot_all(signal: np.ndarray, sample_rate_hz: float, decoded_bits: Optional[str], bitrate_hz: float, psd: Tuple[np.ndarray, np.ndarray]) -> None:
    plt.figure(figsize=(12, 8))
    # 1: waveform
    ax1 = plt.subplot(3, 1, 1)
    time = np.arange(len(signal)) / sample_rate_hz
    ax1.plot(time, signal, lw=1.0)
    ax1.set_title("Raw Signal Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    # 2: decoded bits
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    if decoded_bits:
        step = 1.0 / bitrate_hz
        t = [0.0]
        y = [0.0]
        cur = 0.0
        for b in decoded_bits:
            bit_val = 1.0 if b == "1" else 0.0
            y.extend([bit_val, bit_val])
            t.extend([cur, cur + step])
            cur += step
        ax2.step(t, y[1:] + [y[-1]], where="post")
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_title("Decoded Bits (step)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Bit")
    ax2.grid(True)
    # 3: PSD
    ax3 = plt.subplot(3, 1, 3)
    f, pxx = psd
    ax3.semilogy(f + 1e-9, pxx + 1e-18)
    ax3.set_title("Power Spectral Density (PSD)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD")
    ax3.grid(True, which="both", ls=":")
    plt.tight_layout()


def live_plot(signal: np.ndarray, sample_rate_hz: float, window_seconds: float = 2.0, interval_ms: int = 50) -> None:
    # Simulated live plot of the waveform
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_title("Live Signal Waveform (simulated)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    n = len(signal)
    time = np.arange(n) / sample_rate_hz
    window_samples = int(window_seconds * sample_rate_hz)
    line, = ax.plot([], [], lw=1.2)
    ax.set_xlim(0, window_seconds)
    ax.set_ylim(-0.2, 1.2)

    state = {"idx": 0}

    def update(_):
        start = state["idx"]
        end = min(start + window_samples, n)
        t_win = time[start:end] - time[start] if end > start else []
        y_win = signal[start:end] if end > start else []
        line.set_data(t_win, y_win)
        state["idx"] = end if end < n else 0
        return line,

    FuncAnimation(fig, update, interval=interval_ms, blit=True)
    plt.tight_layout()


# =========================
# High-level flows
# =========================
def run_demo(bits: str, encoding: str, bitrate_hz: float, sample_rate_hz: float, noise_std: float) -> DetectionResult:
    sig = generate_signal(bits, bitrate_hz, sample_rate_hz, encoding, noise_std)
    dom_hz, snr_db, peaks_hz, psd = detect_periodicity(sig, sample_rate_hz)

    decoded: Optional[str] = None
    enc = encoding.lower()
    try:
        if enc == "ook":
            decoded = decode_ook(sig, sample_rate_hz, bitrate_hz)
        elif enc == "pwm":
            decoded = decode_pwm(sig, sample_rate_hz, bitrate_hz)
        elif enc == "manchester":
            decoded = decode_manchester(sig, sample_rate_hz, bitrate_hz)
    except ValueError:
        decoded = None

    errors = compared = None
    ber = accuracy = None
    if decoded is not None:
        errors, compared, ber, accuracy = compute_ber(bits, decoded)

    return DetectionResult(
        encoding=encoding,
        bitrate_hz=bitrate_hz,
        sample_rate_hz=sample_rate_hz,
        noise_std=noise_std,
        dominant_hz=dom_hz,
        snr_db=snr_db,
        peaks_hz=peaks_hz,
        decoded_bits=decoded,
        errors=errors,
        compared_bits=compared,
        ber=ber,
        accuracy_pct=accuracy,
    )


def sweep_noise(bits: str, encodings: List[str], bitrate_hz: float, sample_rate_hz: float, noise_values: List[float]) -> Dict[str, Dict[str, List[float]]]:
    # Returns per-encoding curves: {"OOK": {"noise": [...], "snr_db": [...], "ber": [...]}, ...}
    results: Dict[str, Dict[str, List[float]]] = {}
    for encoding in encodings:
        snr_curve: List[float] = []
        ber_curve: List[float] = []
        for nz in noise_values:
            res = run_demo(bits, encoding, bitrate_hz, sample_rate_hz, nz)
            snr_curve.append(res.snr_db)
            ber_curve.append(res.ber if res.ber is not None else math.nan)
        results[encoding] = {"noise": list(noise_values), "snr_db": snr_curve, "ber": ber_curve}
    return results


def batch_test(num_cases: int, bit_length: int, encodings: List[str], bitrate_hz: float, sample_rate_hz: float, noise_values: List[float], seed: Optional[int] = None) -> List[Dict]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    all_rows: List[Dict] = []
    for _ in range(num_cases):
        bits = generate_random_bits(bit_length)
        curves = sweep_noise(bits, encodings, bitrate_hz, sample_rate_hz, noise_values)
        for encoding, data in curves.items():
            for nz, snr, ber in zip(data["noise"], data["snr_db"], data["ber"]):
                all_rows.append(
                    {
                        "encoding": encoding,
                        "bit_length": bit_length,
                        "bitrate_hz": bitrate_hz,
                        "sample_rate_hz": sample_rate_hz,
                        "noise_std": nz,
                        "snr_db": snr,
                        "ber": ber,
                    }
                )
    return all_rows


def plot_sweep(curves: Dict[str, Dict[str, List[float]]], title_prefix: str = "") -> None:
    # SNR vs Noise
    plt.figure(figsize=(10, 3.5))
    for enc, data in curves.items():
        plt.plot(data["noise"], data["snr_db"], marker="o", label=enc)
    plt.title(f"{title_prefix}SNR vs Noise")
    plt.xlabel("Noise std")
    plt.ylabel("SNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # BER vs Noise
    plt.figure(figsize=(10, 3.5))
    for enc, data in curves.items():
        plt.plot(data["noise"], data["ber"], marker="o", label=enc)
    plt.title(f"{title_prefix}BER vs Noise")
    plt.xlabel("Noise std")
    plt.ylabel("BER")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Safe synthetic covert-channel detector with visualization (no device control).")
    parser.add_argument("--bits", type=str, default="", help="Bitstring like 10110011. If empty and --random-bits used, random bits are generated.")
    parser.add_argument("--random-bits", type=int, default=32, help="If --bits is empty, generate this many random bits.")
    parser.add_argument("--encoding", type=str, choices=["OOK", "Manchester", "PWM"], default="Manchester")
    parser.add_argument("--bitrate", type=float, default=25.0, help="Bitrate in bits per second")
    parser.add_argument("--sample-rate", type=float, default=2000.0, help="Samples per second")
    parser.add_argument("--noise-std", type=float, default=0.05)

    parser.add_argument("--plot", action="store_true", help="Show waveform, decoded bits, and PSD plots")
    parser.add_argument("--live", action="store_true", help="Show a simulated live waveform plot")
    parser.add_argument("--save-signal", type=str, default="", help="Path to save signal (.npy, .npz, or .csv)")
    parser.add_argument("--load-signal", type=str, default="", help="Path to load signal (.npy, .npz, or .csv)")
    parser.add_argument("--save-result", type=str, default="", help="Path to save result JSON")

    # Noise sweep / batch testing
    parser.add_argument("--sweep-noise", type=str, default="", help="Comma-separated noise stds, e.g. 0,0.02,0.05,0.1,0.2")
    parser.add_argument("--sweep-encodings", type=str, default="OOK,Manchester,PWM", help="Encodings to compare, comma-separated")
    parser.add_argument("--batch", type=int, default=0, help="If >0, run batch tests with this many random bitstreams per encoding/noise")
    parser.add_argument("--bitlen", type=int, default=64, help="Bit length for random tests")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--export-csv", type=str, default="", help="Path to export batch results as CSV")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load or generate signal
    meta: Dict = {}
    if args.load_signal:
        signal, sample_rate_hz, meta = load_signal(args.load_signal, fallback_sample_rate_hz=args.sample_rate)
        bits = sanitize_bits(args.bits) or meta.get("bits", "")
        encoding = meta.get("encoding", args.encoding)
        bitrate_hz = float(meta.get("bitrate_hz", args.bitrate))
        noise_std = float(meta.get("noise_std", args.noise_std))
    else:
        bits = sanitize_bits(args.bits) or generate_random_bits(args.random_bits)
        encoding = args.encoding
        bitrate_hz = args.bitrate
        sample_rate_hz = args.sample_rate
        noise_std = args.noise_std
        signal = generate_signal(bits, bitrate_hz, sample_rate_hz, encoding, noise_std)

    dom_hz, snr_db, peaks_hz, psd = detect_periodicity(signal, sample_rate_hz)

    # Decode
    decoded: Optional[str] = None
    try:
        if encoding.lower() == "ook":
            decoded = decode_ook(signal, sample_rate_hz, bitrate_hz)
        elif encoding.lower() == "pwm":
            decoded = decode_pwm(signal, sample_rate_hz, bitrate_hz)
        elif encoding.lower() == "manchester":
            decoded = decode_manchester(signal, sample_rate_hz, bitrate_hz)
    except ValueError:
        decoded = None

    errors = compared = None
    ber = accuracy = None
    if decoded is not None:
        errors, compared, ber, accuracy = compute_ber(bits, decoded)

    result = DetectionResult(
        encoding=encoding,
        bitrate_hz=bitrate_hz,
        sample_rate_hz=sample_rate_hz,
        noise_std=noise_std,
        dominant_hz=dom_hz,
        snr_db=snr_db,
        peaks_hz=peaks_hz,
        decoded_bits=decoded,
        errors=errors,
        compared_bits=compared,
        ber=ber,
        accuracy_pct=accuracy,
    )

    # Print concise summary
    print(f"Encoding: {result.encoding}")
    print(f"Bitrate: {result.bitrate_hz} bps  |  Sample rate: {result.sample_rate_hz} Hz  |  Noise std: {result.noise_std}")
    print(f"Dominant frequency: {result.dominant_hz:.2f} Hz  |  SNR: {result.snr_db:.1f} dB")
    if result.decoded_bits is not None:
        print(f"Decoded bits: {result.decoded_bits[:64]}{'...' if len(result.decoded_bits or '') > 64 else ''}")
    if result.ber is not None:
        print(f"BER: {result.ber:.4f}  |  Accuracy: {result.accuracy_pct:.2f}%  |  Errors/Compared: {result.errors}/{result.compared_bits}")

    # Save signal if requested
    if args.save_signal:
        metadata = {
            "bits": bits,
            "encoding": encoding,
            "bitrate_hz": bitrate_hz,
            "noise_std": noise_std,
        }
        save_signal(args.save_signal, signal, sample_rate_hz, metadata=metadata)
        print(f"Saved signal to {os.path.abspath(args.save_signal)}")

    # Save result JSON if requested
    if args.save_result:
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Saved result to {os.path.abspath(args.save_result)}")

    # Noise sweeps / batch
    if args.sweep_noise or args.batch > 0:
        encodings = [e.strip() for e in args.sweep_encodings.split(",") if e.strip()]
        if args.sweep_noise:
            noise_values = [float(x) for x in args.sweep_noise.split(",")]
        else:
            noise_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]

        if args.batch > 0:
            rows = batch_test(args.batch, args.bitlen, encodings, bitrate_hz, sample_rate_hz, noise_values, seed=args.seed)
            if args.export_csv:
                import csv
                with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Exported batch results to {os.path.abspath(args.export_csv)}")
            # Aggregate mean BER/SNR per encoding/noise for plotting
            agg: Dict[str, Dict[str, List[float]]] = {enc: {"noise": noise_values, "snr_db": [], "ber": []} for enc in encodings}
            for enc in encodings:
                for nz in noise_values:
                    snrs = [r["snr_db"] for r in rows if r["encoding"] == enc and r["noise_std"] == nz]
                    bers = [r["ber"] for r in rows if r["encoding"] == enc and r["noise_std"] == nz]
                    agg[enc]["snr_db"].append(float(np.nanmean(snrs)) if snrs else math.nan)
                    agg[enc]["ber"].append(float(np.nanmean(bers)) if bers else math.nan)
            plot_sweep(agg, title_prefix="Batch mean ")
        else:
            curves = sweep_noise(bits, encodings, bitrate_hz, sample_rate_hz, noise_values)
            plot_sweep(curves)

    # Plots
    if args.plot:
        plot_all(signal, sample_rate_hz, result.decoded_bits, bitrate_hz, psd)
        if args.live:
            live_plot(signal, sample_rate_hz)
        plt.show()
    elif args.live:
        live_plot(signal, sample_rate_hz)
        plt.show()


if __name__ == "__main__":
    main()