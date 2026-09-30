from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# ============================================================
# 1. 데이터셋 경로
# ============================================================

DATASETS = {
    "set1": "data/test_set_1_1ms.csv",
    "set2": "data/test_set_2_1ms.csv",
    "set3": "data/test_set_3_1ms.csv",
}


# ============================================================
# 2. 샘플링 주파수
#
# 1 ms 간격
# → 0.001초
# → 1 / 0.001 = 1000 Hz
# ============================================================

FS = 1000


# ============================================================
# 3. 시간영역 특징 추출
# ============================================================


def extract_time_features(signal):

    signal = np.asarray(signal, dtype=float)

    # RMS
    rms = np.sqrt(np.mean(signal**2))

    # 첨도
    # fisher=False → 정규분포 첨도 = 3
    kurt = kurtosis(signal, fisher=False, bias=False)

    # 최댓값 / 최솟값
    max_value = np.max(signal)
    min_value = np.min(signal)

    # 절대 Peak
    peak = np.max(np.abs(signal))

    # 파고율
    if rms != 0:
        crest_factor = peak / rms
    else:
        crest_factor = 0

    # 표준편차
    std = np.std(signal)

    # Peak-to-Peak
    peak_to_peak = max_value - min_value

    return {
        "rms": rms,
        "kurtosis": kurt,
        "max": max_value,
        "min": min_value,
        "crest_factor": crest_factor,
        "std": std,
        "peak_to_peak": peak_to_peak,
    }


# ============================================================
# 4. 주파수영역 특징 추출
# ============================================================


def extract_frequency_features(signal, fs):

    signal = np.asarray(signal, dtype=float)

    n = len(signal)

    # 너무 짧은 신호 방지
    if n < 2:

        return {
            "dominant_freq": np.nan,
            "spectral_centroid": np.nan,
            "spectral_energy": np.nan,
            "spectral_entropy": np.nan,
        }

    # --------------------------------------------------------
    # 평균 제거
    #
    # DC 성분이 dominant frequency로 잡히는 것을 방지
    # --------------------------------------------------------

    signal_centered = signal - np.mean(signal)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft_values = np.fft.rfft(signal_centered)

    frequencies = np.fft.rfftfreq(n, d=1 / fs)

    # --------------------------------------------------------
    # Magnitude Spectrum
    # --------------------------------------------------------

    magnitude = np.abs(fft_values)

    # DC(0 Hz) 제거
    if len(magnitude) > 0:

        magnitude[0] = 0

    # ========================================================
    # 1. Dominant Frequency
    #
    # 가장 강한 진동 주파수
    # ========================================================

    dominant_index = np.argmax(magnitude)

    dominant_freq = frequencies[dominant_index]

    # ========================================================
    # 2. Spectral Centroid
    #
    # 주파수 에너지의 중심
    # ========================================================

    magnitude_sum = np.sum(magnitude)

    if magnitude_sum != 0:

        spectral_centroid = np.sum(frequencies * magnitude) / magnitude_sum

    else:

        spectral_centroid = 0

    # ========================================================
    # 3. Spectral Energy
    #
    # FFT 전체 에너지
    # ========================================================

    spectral_energy = np.sum(magnitude**2)

    # ========================================================
    # 4. Spectral Entropy
    #
    # 주파수 에너지가
    # 얼마나 퍼져있는지를 표현
    # ========================================================

    power = magnitude**2

    power_sum = np.sum(power)

    if power_sum != 0:

        probability = power / power_sum

        probability = probability[probability > 0]

        spectral_entropy = -np.sum(probability * np.log2(probability))

    else:

        spectral_entropy = 0

    return {
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy,
    }


# ============================================================
# 5. 데이터셋 하나 처리
# ============================================================


def process_dataset(file_path, dataset_name):

    file_path = Path(file_path)

    # --------------------------------------------------------
    # 데이터 불러오기
    # --------------------------------------------------------

    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # 시간순 정렬
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ========================================================
    # 6. 시간 차이 계산
    # ========================================================

    df["time_diff"] = df["timestamp"].diff()

    # ========================================================
    # 7. Segment 분리
    #
    # 내부 데이터는 약 1ms 간격
    # 1초 이상 벌어지면 새로운 측정 Segment
    # ========================================================

    GAP_THRESHOLD = pd.Timedelta(seconds=1)

    df["segment"] = (df["time_diff"] > GAP_THRESHOLD).cumsum()

    print("\n==============================")
    print(dataset_name)
    print("==============================")

    print("원본 행 수:", len(df))

    print("측정 Segment 수:", df["segment"].nunique())

    # ========================================================
    # 8. Segment 크기 확인
    # ========================================================

    segment_sizes = df.groupby("segment").size()

    print("\nSegment 크기 분포")

    print(segment_sizes.value_counts().sort_index())

    # ========================================================
    # 9. 센서 컬럼 선택
    # ========================================================

    exclude_cols = {"timestamp", "time_diff", "segment"}

    sensor_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in exclude_cols
    ]

    print("\n사용 센서 컬럼")

    print(sensor_cols)

    # ========================================================
    # 10. Segment별 특징 추출
    # ========================================================

    result_rows = []

    for segment_id, group in df.groupby("segment"):

        row = {
            "dataset": dataset_name,
            "segment": segment_id,
            "start_time": group["timestamp"].iloc[0],
            "end_time": group["timestamp"].iloc[-1],
            "n_samples": len(group),
        }

        # ----------------------------------------------------
        # 각 센서별 특징 추출
        # ----------------------------------------------------

        for col in sensor_cols:

            signal = group[col].dropna().values

            if len(signal) == 0:
                continue

            # =================================================
            # 시간영역 특징
            # =================================================

            time_features = extract_time_features(signal)

            for feature_name, value in time_features.items():

                column_name = f"{col}_" f"{feature_name}"

                row[column_name] = value

            # =================================================
            # 주파수영역 특징
            # =================================================

            frequency_features = extract_frequency_features(signal, FS)

            for feature_name, value in frequency_features.items():

                column_name = f"{col}_" f"{feature_name}"

                row[column_name] = value

        result_rows.append(row)

    # ========================================================
    # 11. 특징 DataFrame 생성
    # ========================================================

    feature_df = pd.DataFrame(result_rows)

    return feature_df


# ============================================================
# 12. 세 데이터셋 각각 처리
# ============================================================

for dataset_name, file_path in DATASETS.items():

    feature_df = process_dataset(file_path=file_path, dataset_name=dataset_name)

    # ========================================================
    # 13. 세트별 별도 저장
    # ========================================================

    output_path = f"data/" f"{dataset_name}_features.csv"

    feature_df.to_csv(output_path, index=False)

    print("\n저장 완료")

    print(output_path)

    print("특징 데이터 Shape:", feature_df.shape)
