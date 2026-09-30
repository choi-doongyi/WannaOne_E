# ============================================================
# 09_FFT_Analysis.py
#
# 선택과제 - FFT 기반 주파수 영역 분석
#
# 1. 열화 전 / 열화 후 FFT 스펙트럼 비교
# 2. Spectral Energy 시간 추세
# 3. 증가한 주파수 대역 분석
#
# 주의:
# - 샘플링 주파수 = 1000 Hz
# - Nyquist Frequency = 500 Hz
# - 축 회전수 / 베어링 형상 정보가 없으므로
#   BPFO, BPFI, BSF, FTF 등 결함 종류는 단정하지 않음
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 한글 폰트
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 기본 설정
# ============================================================

FS = 1000

GAP_THRESHOLD = pd.Timedelta(seconds=1)

# 열화 전 / 후 각각 사용할 segment 개수
COMPARE_SEGMENTS = 10


# ============================================================
# 3. Raw Data
# ============================================================

RAW_DATASETS = {
    "set1": "data/test_set_1_1ms.csv",
    "set2": "data/test_set_2_1ms.csv",
    "set3": "data/test_set_3_1ms.csv",
}


# ============================================================
# 4. Feature Data
#
# Spectral Energy 추세 확인용
# ============================================================

FEATURE_DATASETS = {
    "set1": "data/model_ready/set1_model_ready.csv",
    "set2": "data/model_ready/set2_model_ready.csv",
    "set3": "data/model_ready/set3_model_ready.csv",
}


# ============================================================
# 5. 열화 시작 결과
# ============================================================

DEGRADATION_SUMMARY_PATH = "results/moving_zscore/" "moving_zscore_summary.csv"


# ============================================================
# 6. 저장 폴더
# ============================================================

SAVE_DIR = "results/fft_analysis"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 7. 열화 시작 결과 불러오기
# ============================================================

degradation_summary = pd.read_csv(
    DEGRADATION_SUMMARY_PATH, parse_dates=["degradation_start"]
)


# ============================================================
# 8. FFT 계산 함수
# ============================================================


def calculate_fft(signal):

    signal = np.asarray(signal, dtype=float)

    # --------------------------------------------------------
    # DC 성분 제거
    # --------------------------------------------------------

    signal = signal - np.mean(signal)

    n = len(signal)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft_values = np.fft.rfft(signal)

    frequencies = np.fft.rfftfreq(n, d=1 / FS)

    # --------------------------------------------------------
    # Amplitude Spectrum
    # --------------------------------------------------------

    amplitude = np.abs(fft_values) / n * 2

    # DC는 제거
    if len(amplitude) > 0:

        amplitude[0] = 0

    return (frequencies, amplitude)


# ============================================================
# 9. 주파수 대역 에너지 계산
# ============================================================


def calculate_band_energy(frequencies, amplitude, band_start, band_end):

    mask = (frequencies >= band_start) & (frequencies < band_end)

    if mask.sum() == 0:

        return 0

    energy = np.sum(amplitude[mask] ** 2)

    return energy


# ============================================================
# 10. 주파수 Band 정의
#
# 0 ~ 500 Hz
# 50 Hz 단위
# ============================================================

FREQ_BANDS = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
    (300, 350),
    (350, 400),
    (400, 450),
    (450, 500),
]


# ============================================================
# 11. 전체 결과 저장용
# ============================================================

band_results = []

summary_results = []


# ============================================================
# 12. 데이터셋별 분석
# ============================================================

for dataset_name in RAW_DATASETS.keys():

    print("\n" + "=" * 80)

    print(dataset_name)

    print("=" * 80)

    # ========================================================
    # 13. 열화 시작 시점
    # ========================================================

    degradation_row = degradation_summary[
        degradation_summary["dataset"] == dataset_name
    ]

    degradation_time = degradation_row["degradation_start"].iloc[0]

    print("\n열화 시작 시점:")

    print(degradation_time)

    # ========================================================
    # 14. Raw Data 불러오기
    # ========================================================

    raw_path = RAW_DATASETS[dataset_name]

    raw_df = pd.read_csv(raw_path, parse_dates=["timestamp"])

    raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

    # ========================================================
    # 15. 채널 선택
    # ========================================================

    channels = [col for col in raw_df.columns if col.startswith("channel_")]

    print("\n채널:")

    print(channels)

    # ========================================================
    # 16. Segment 생성
    #
    # 1초 이상 시간 차이 발생 시
    # 새로운 측정 session
    # ========================================================

    raw_df["time_diff"] = raw_df["timestamp"].diff()

    raw_df["segment"] = (raw_df["time_diff"] > GAP_THRESHOLD).cumsum()

    # ========================================================
    # 17. Segment 정보
    # ========================================================

    segment_info = (
        raw_df.groupby("segment")
        .agg(
            start_time=("timestamp", "first"),
            end_time=("timestamp", "last"),
            n_samples=("timestamp", "size"),
        )
        .reset_index()
    )

    # ========================================================
    # 18. 대표 Sample Size
    # ========================================================

    main_sample_size = segment_info["n_samples"].mode().iloc[0]

    print("\n대표 Segment Sample Size:")

    print(main_sample_size)

    # ========================================================
    # 19. 동일 길이 Segment만 사용
    # ========================================================

    valid_segments = segment_info[segment_info["n_samples"] == main_sample_size].copy()

    # ========================================================
    # 20. 열화 전 / 후 Segment 선택
    #
    # 열화 직전 10개
    # 열화 직후 10개
    # ========================================================

    before_segments = valid_segments[
        valid_segments["start_time"] < degradation_time
    ].tail(COMPARE_SEGMENTS)

    after_segments = valid_segments[
        valid_segments["start_time"] >= degradation_time
    ].head(COMPARE_SEGMENTS)

    print("\n열화 전 Segment 수:")

    print(len(before_segments))

    print("열화 후 Segment 수:")

    print(len(after_segments))

    # ========================================================
    # 21. Dataset 저장 폴더
    # ========================================================

    dataset_dir = os.path.join(SAVE_DIR, dataset_name)

    os.makedirs(dataset_dir, exist_ok=True)

    # ========================================================
    # 22. 채널별 FFT 분석
    # ========================================================

    channel_before_spectra = {}

    channel_after_spectra = {}

    for channel in channels:

        before_fft_list = []

        after_fft_list = []

        # ----------------------------------------------------
        # 열화 전 FFT
        # ----------------------------------------------------

        for segment_id in before_segments["segment"]:

            segment_signal = raw_df[raw_df["segment"] == segment_id][channel].values

            frequencies, amplitude = calculate_fft(segment_signal)

            before_fft_list.append(amplitude)

        # ----------------------------------------------------
        # 열화 후 FFT
        # ----------------------------------------------------

        for segment_id in after_segments["segment"]:

            segment_signal = raw_df[raw_df["segment"] == segment_id][channel].values

            frequencies_after, amplitude = calculate_fft(segment_signal)

            after_fft_list.append(amplitude)

        # ----------------------------------------------------
        # FFT 평균
        # ----------------------------------------------------

        before_fft_mean = np.mean(before_fft_list, axis=0)

        after_fft_mean = np.mean(after_fft_list, axis=0)

        channel_before_spectra[channel] = before_fft_mean

        channel_after_spectra[channel] = after_fft_mean

        # ====================================================
        # 23. 채널별 FFT 그래프
        # ====================================================

        plt.figure(figsize=(14, 5))

        plt.plot(frequencies, before_fft_mean, label="열화 전")

        plt.plot(frequencies, after_fft_mean, label="열화 후")

        plt.title(f"{dataset_name} - " f"{channel} FFT 비교")

        plt.xlabel("Frequency (Hz)")

        plt.ylabel("Amplitude")

        plt.xlim(0, FS / 2)

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(dataset_dir, f"{channel}_fft_before_after.png"),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # ====================================================
        # 24. 주파수 대역 에너지 분석
        # ====================================================

        for band_start, band_end in FREQ_BANDS:

            before_energy = calculate_band_energy(
                frequencies, before_fft_mean, band_start, band_end
            )

            after_energy = calculate_band_energy(
                frequencies, after_fft_mean, band_start, band_end
            )

            # ------------------------------------------------
            # 변화율
            # ------------------------------------------------

            if before_energy > 0:

                change_ratio = after_energy / before_energy

            else:

                change_ratio = np.nan

            band_results.append(
                {
                    "dataset": dataset_name,
                    "channel": channel,
                    "band_start": band_start,
                    "band_end": band_end,
                    "before_energy": before_energy,
                    "after_energy": after_energy,
                    "energy_ratio": change_ratio,
                }
            )

    # ========================================================
    # 25. 전체 채널 평균 FFT
    #
    # 발표용 대표 그래프
    # ========================================================

    all_before = np.mean(np.array(list(channel_before_spectra.values())), axis=0)

    all_after = np.mean(np.array(list(channel_after_spectra.values())), axis=0)

    plt.figure(figsize=(14, 5))

    plt.plot(frequencies, all_before, label="열화 전 평균")

    plt.plot(frequencies, all_after, label="열화 후 평균")

    plt.title(f"{dataset_name} 전체 채널 평균 FFT 비교")

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Amplitude")

    plt.xlim(0, FS / 2)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_dir, "all_channels_fft_before_after.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 26. Spectral Energy 추세 분석
    # ========================================================

    feature_path = FEATURE_DATASETS[dataset_name]

    feature_df = pd.read_csv(feature_path, parse_dates=["start_time", "end_time"])

    feature_df = feature_df.sort_values("start_time").reset_index(drop=True)

    spectral_cols = [
        col for col in feature_df.columns if col.endswith("_spectral_energy")
    ]

    print("\nSpectral Energy 특징 개수:")

    print(len(spectral_cols))

    # ========================================================
    # 27. 채널별 Spectral Energy 그래프
    # ========================================================

    for col in spectral_cols:

        plt.figure(figsize=(14, 5))

        plt.plot(feature_df["start_time"], feature_df[col], label=col)

        plt.axvline(x=degradation_time, linestyle="--", label="열화 시작")

        plt.title(f"{dataset_name} - " f"{col} 추세")

        plt.xlabel("Time")

        plt.ylabel("Spectral Energy")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(dataset_dir, f"{col}_trend.png"), dpi=300, bbox_inches="tight"
        )

        plt.close()

    # ========================================================
    # 28. 전체 채널 평균 Spectral Energy
    # ========================================================

    feature_df["mean_spectral_energy"] = feature_df[spectral_cols].mean(axis=1)

    plt.figure(figsize=(14, 5))

    plt.plot(
        feature_df["start_time"],
        feature_df["mean_spectral_energy"],
        label="전체 채널 평균 Spectral Energy",
    )

    plt.axvline(x=degradation_time, linestyle="--", label="열화 시작")

    plt.title(f"{dataset_name} 평균 Spectral Energy 추세")

    plt.xlabel("Time")

    plt.ylabel("Mean Spectral Energy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_dir, "mean_spectral_energy_trend.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 29. 데이터셋별 Band 결과
    # ========================================================

    current_band_df = pd.DataFrame(
        [row for row in band_results if row["dataset"] == dataset_name]
    )

    current_band_df.to_csv(
        os.path.join(dataset_dir, "frequency_band_energy.csv"), index=False
    )

    # ========================================================
    # 30. 대역별 평균 증가율
    # ========================================================

    band_summary = current_band_df.groupby(
        ["band_start", "band_end"], as_index=False
    ).agg(
        mean_before_energy=("before_energy", "mean"),
        mean_after_energy=("after_energy", "mean"),
        mean_energy_ratio=("energy_ratio", "mean"),
    )

    band_summary["band"] = (
        band_summary["band_start"].astype(str)
        + "-"
        + band_summary["band_end"].astype(str)
        + " Hz"
    )

    band_summary.to_csv(
        os.path.join(dataset_dir, "frequency_band_summary.csv"), index=False
    )

    # ========================================================
    # 31. 대역별 에너지 증가율 그래프
    # ========================================================

    plt.figure(figsize=(12, 6))

    plt.bar(band_summary["band"], band_summary["mean_energy_ratio"])

    plt.axhline(y=1, linestyle="--", label="변화 없음")

    plt.title(f"{dataset_name} 주파수 대역별 에너지 변화율")

    plt.xlabel("Frequency Band")

    plt.ylabel("After / Before Energy Ratio")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(dataset_dir, "frequency_band_energy_ratio.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 32. 가장 크게 증가한 주파수 대역
    # ========================================================

    valid_summary = band_summary.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mean_energy_ratio"]
    )

    top_band = valid_summary.sort_values("mean_energy_ratio", ascending=False).iloc[0]

    print("\n가장 크게 증가한 주파수 대역:")

    print(top_band["band"])

    print("에너지 변화율:")

    print(top_band["mean_energy_ratio"])

    summary_results.append(
        {
            "dataset": dataset_name,
            "degradation_start": degradation_time,
            "sample_size": main_sample_size,
            "before_segment_count": len(before_segments),
            "after_segment_count": len(after_segments),
            "top_frequency_band": top_band["band"],
            "top_energy_ratio": top_band["mean_energy_ratio"],
        }
    )


# ============================================================
# 33. 전체 Band 결과 저장
# ============================================================

band_result_df = pd.DataFrame(band_results)


band_result_df.to_csv(
    os.path.join(SAVE_DIR, "all_frequency_band_energy.csv"), index=False
)


# ============================================================
# 34. 전체 요약 저장
# ============================================================

summary_df = pd.DataFrame(summary_results)


summary_df.to_csv(os.path.join(SAVE_DIR, "fft_analysis_summary.csv"), index=False)


# ============================================================
# 35. 최종 출력
# ============================================================

print("\n" + "=" * 80)


print("FFT 선택과제 최종 결과")


print("=" * 80)


print(summary_df)


print("\n" + "=" * 80)


print("09 FFT 선택과제 분석 완료")


print("=" * 80)

# “FFT 분석 결과, 열화 이후 특정 주파수 대역의 에너지가 증가했으며,
# 특히 set2의 450–500 Hz 대역에서 약 26.9% 증가가 확인되었다.”
# 베어링 형상/회전수 정보가 없어서 BPFO/BPFI 같은 결함 종류를 특정하지는 않는다.
