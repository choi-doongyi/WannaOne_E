from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from src.config import DATA_DIR, FS, GAP_THRESHOLD, RAW_DATASETS

def extract_time_features(signal):
    signal = np.asarray(signal, dtype=float)
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal, fisher=False, bias=False)
    max_value = np.max(signal)
    min_value = np.min(signal)
    peak = np.max(np.abs(signal))
    return {
        "rms": rms,
        "kurtosis": kurt,
        "max": max_value,
        "min": min_value,
        "crest_factor": peak / rms if rms != 0 else 0,
        "std": np.std(signal),
        "peak_to_peak": max_value - min_value,
    }

def extract_frequency_features(signal, fs=FS):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 2:
        return {"dominant_freq": np.nan, "spectral_centroid": np.nan, "spectral_energy": np.nan, "spectral_entropy": np.nan}
    signal_centered = signal - np.mean(signal)
    fft_values = np.fft.rfft(signal_centered)
    frequencies = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(fft_values)
    if len(magnitude) > 0:
        magnitude[0] = 0
    dominant_freq = frequencies[np.argmax(magnitude)]
    magnitude_sum = np.sum(magnitude)
    spectral_centroid = np.sum(frequencies * magnitude) / magnitude_sum if magnitude_sum != 0 else 0
    spectral_energy = np.sum(magnitude**2)
    power = magnitude**2
    power_sum = np.sum(power)
    if power_sum != 0:
        probability = power / power_sum
        probability = probability[probability > 0]
        spectral_entropy = -np.sum(probability * np.log2(probability))
    else:
        spectral_entropy = 0
    return {"dominant_freq": dominant_freq, "spectral_centroid": spectral_centroid, "spectral_energy": spectral_energy, "spectral_entropy": spectral_entropy}

def process_dataset(file_path, dataset_name):
    df = pd.read_csv(Path(file_path), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_diff"] = df["timestamp"].diff()
    df["segment"] = (df["time_diff"] > GAP_THRESHOLD).cumsum()
    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)
    print("원본 행 수:", len(df))
    print("측정 Segment 수:", df["segment"].nunique())
    segment_sizes = df.groupby("segment").size()
    print("\nSegment 크기 분포")
    print(segment_sizes.value_counts().sort_index())
    exclude_cols = {"timestamp", "time_diff", "segment"}
    sensor_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]
    print("\n사용 센서 컬럼")
    print(sensor_cols)
    result_rows = []
    for segment_id, group in df.groupby("segment", sort=True):
        row = {"dataset": dataset_name, "segment": segment_id, "start_time": group["timestamp"].iloc[0], "end_time": group["timestamp"].iloc[-1], "n_samples": len(group)}
        for col in sensor_cols:
            signal = group[col].dropna().values
            if len(signal) == 0:
                continue
            for feature_name, value in extract_time_features(signal).items():
                row[f"{col}_{feature_name}"] = value
            for feature_name, value in extract_frequency_features(signal).items():
                row[f"{col}_{feature_name}"] = value
        result_rows.append(row)
    return pd.DataFrame(result_rows)

def run():
    for dataset_name, file_path in RAW_DATASETS.items():
        feature_df = process_dataset(file_path, dataset_name)
        output_path = DATA_DIR / f"{dataset_name}_features.csv"
        feature_df.to_csv(output_path, index=False)
        print("\n저장 완료:", output_path)
        print("특징 데이터 Shape:", feature_df.shape)

if __name__ == "__main__":
    run()
