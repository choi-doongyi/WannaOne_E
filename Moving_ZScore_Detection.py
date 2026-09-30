# ============================================================
# 05_Moving_ZScore_Detection.py
# 베어링 필수 특징 기반 Robust Moving Z-score 열화 시작 탐지
#
# 기준:
# - 초기 30%를 정상 기준 구간으로 사용
# - 평균 / 표준편차 대신 중앙값 / MAD 사용
# - Robust Z-score >= 3
# - 서로 다른 채널 2개 이상
# - 3개 segment 연속
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 한글 폰트
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 데이터 경로
# ============================================================

DATASETS = {
    "set1": "data/model_ready/set1_model_ready.csv",
    "set2": "data/model_ready/set2_model_ready.csv",
    "set3": "data/model_ready/set3_model_ready.csv",
}


# ============================================================
# 3. 저장 폴더
# ============================================================

SAVE_DIR = "results/moving_zscore"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 4. 분석 기준
# ============================================================

# 초기 30%를 정상 기준 구간으로 사용
BASELINE_RATIO = 0.30

# Robust Z-score 이상 기준
Z_THRESHOLD = 3.0

# 최근 5개 segment 평균
ROLLING_WINDOW = 5

# 서로 다른 채널 2개 이상 이상
MIN_CHANNELS = 2

# 3개 segment 연속
CONSECUTIVE_COUNT = 3


# ============================================================
# 5. Robust Z-score 보정 상수
#
# 정규분포에서 MAD를 표준편차와 유사한 스케일로 맞추기 위해 사용
# ============================================================

ROBUST_SCALE = 0.6745


# ============================================================
# 6. 채널 번호 추출 함수
# ============================================================


def get_channel_number(column_name):

    match = re.search(r"channel_(\d+)_", column_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 7. 연속 이상 시작점 탐색 함수
# ============================================================


def find_consecutive_start(series, consecutive_count):

    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()

    detected = rolling_count[rolling_count >= consecutive_count]

    if len(detected) == 0:
        return None

    end_idx = detected.index[0]

    start_idx = end_idx - consecutive_count + 1

    return start_idx


# ============================================================
# 8. MAD 계산 함수
#
# MAD = median(|x - median(x)|)
# ============================================================


def calculate_mad(series):

    median_value = series.median()

    mad_value = (series - median_value).abs().median()

    return (median_value, mad_value)


# ============================================================
# 9. 전체 결과 저장용
# ============================================================

summary_results = []

cause_results = []


# ============================================================
# 10. 데이터셋별 실행
# ============================================================

for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 70)

    print(dataset_name)

    print("=" * 70)

    # ========================================================
    # 11. 데이터 불러오기
    # ========================================================

    df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])

    df = df.sort_values("start_time").reset_index(drop=True)

    print("\n데이터 Shape:")

    print(df.shape)

    # ========================================================
    # 12. 필수 특징 선택
    #
    # RMS
    # Kurtosis
    # Max
    # Crest Factor
    # ========================================================

    target_cols = [
        col
        for col in df.columns
        if (
            col.endswith("_rms")
            or col.endswith("_kurtosis")
            or col.endswith("_max")
            or col.endswith("_crest_factor")
        )
    ]

    print("\n필수 특징 개수:")

    print(len(target_cols))

    # ========================================================
    # 13. 초기 정상 구간
    # ========================================================

    baseline_end = int(len(df) * BASELINE_RATIO)

    baseline = df.iloc[:baseline_end].copy()

    print("\nBaseline 데이터 개수:")

    print(len(baseline))

    print("\nBaseline 종료 시점:")

    print(baseline["start_time"].iloc[-1])

    # ========================================================
    # 14. Robust Moving Z-score 계산
    #
    # 1) 최근 5개 segment Moving Average
    # 2) 초기 정상구간 Moving Average 계산
    # 3) 정상구간의 Median / MAD 계산
    # 4) Robust Z-score 계산
    #
    # Robust Z =
    # 0.6745 * (x - median) / MAD
    # ========================================================

    moving_data = {}

    zscore_data = {}

    flag_data = {}

    baseline_stats = []

    for col in target_cols:

        # ----------------------------------------------------
        # 전체 데이터 Moving Average
        # ----------------------------------------------------

        moving_series = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

        moving_col = f"{col}_moving"

        moving_data[moving_col] = moving_series

        # ----------------------------------------------------
        # Baseline Moving Average
        # ----------------------------------------------------

        baseline_moving = (
            baseline[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        # ----------------------------------------------------
        # Median / MAD
        # ----------------------------------------------------

        baseline_median, baseline_mad = calculate_mad(baseline_moving)

        # ----------------------------------------------------
        # MAD가 0인 경우 대체 처리
        #
        # 값이 거의 일정한 특징이면 MAD가 0이 될 수 있음
        # 이 경우 baseline의 IQR 기반 scale을 보조로 사용
        # ----------------------------------------------------

        if pd.isna(baseline_mad) or baseline_mad == 0:

            q1 = baseline_moving.quantile(0.25)

            q3 = baseline_moving.quantile(0.75)

            iqr = q3 - q1

            # IQR도 0이면 해당 특징은 변화가 거의 없음
            if pd.isna(iqr) or iqr == 0:

                robust_z = pd.Series(0.0, index=df.index)

            else:

                # 정규분포 기준으로
                # IQR / 1.349 ≈ sigma
                robust_scale = iqr / 1.349

                robust_z = (moving_series - baseline_median) / robust_scale

        else:

            robust_z = ROBUST_SCALE * (moving_series - baseline_median) / baseline_mad

        # ----------------------------------------------------
        # 기존 06번 호환을 위해 컬럼명은 _zscore 유지
        # ----------------------------------------------------

        z_col = f"{col}_zscore"

        zscore_data[z_col] = robust_z

        # ----------------------------------------------------
        # Robust Z >= 3
        # ----------------------------------------------------

        flag_col = f"{col}_flag"

        flag_data[flag_col] = robust_z >= Z_THRESHOLD

        # ----------------------------------------------------
        # Baseline 통계 저장
        # ----------------------------------------------------

        baseline_stats.append(
            {
                "feature": col,
                "baseline_median": baseline_median,
                "baseline_mad": baseline_mad,
            }
        )

    # ========================================================
    # 15. 계산된 컬럼 한 번에 추가
    #
    # PerformanceWarning 방지
    # ========================================================

    moving_df = pd.DataFrame(moving_data)

    zscore_df = pd.DataFrame(zscore_data)

    flag_df = pd.DataFrame(flag_data)

    df = pd.concat(
        [
            df,
            moving_df,
            zscore_df,
            flag_df,
        ],
        axis=1,
    )

    # ========================================================
    # 16. Baseline 통계 저장
    # ========================================================

    baseline_stats_df = pd.DataFrame(baseline_stats)

    # ========================================================
    # 17. 채널 목록
    # ========================================================

    channels = sorted(
        set(
            get_channel_number(col)
            for col in target_cols
            if get_channel_number(col) is not None
        )
    )

    print("\n채널:")

    print(channels)

    # ========================================================
    # 18. 채널별 이상 여부
    #
    # 한 채널의
    # RMS / Kurtosis / Max / Crest Factor 중
    # 하나라도 Robust Z >= 3
    # ========================================================

    channel_flag_data = {}

    for channel in channels:

        current_flags = [
            col for col in flag_df.columns if col.startswith(f"channel_{channel}_")
        ]

        channel_flag_col = f"channel_{channel}_abnormal"

        channel_flag_data[channel_flag_col] = flag_df[current_flags].any(axis=1)

    channel_flag_df = pd.DataFrame(channel_flag_data)

    df = pd.concat([df, channel_flag_df], axis=1)

    # ========================================================
    # 19. 동시에 이상인 채널 수
    # ========================================================

    df["abnormal_channel_count"] = channel_flag_df.sum(axis=1)

    # ========================================================
    # 20. 열화 후보 판정
    #
    # 서로 다른 채널 2개 이상
    # ========================================================

    df["degradation_candidate"] = df["abnormal_channel_count"] >= MIN_CHANNELS

    # ========================================================
    # 21. Baseline 내부는 열화 판정 제외
    # ========================================================

    df.loc[: baseline_end - 1, "degradation_candidate"] = False

    # ========================================================
    # 22. 연속 조건 적용
    # ========================================================

    degradation_idx = find_consecutive_start(
        df["degradation_candidate"], CONSECUTIVE_COUNT
    )

    # ========================================================
    # 23. 열화 시작 시점 계산
    # ========================================================

    if degradation_idx is not None:

        degradation_time = df.loc[degradation_idx, "start_time"]

        degradation_position_pct = degradation_idx / len(df) * 100

    else:

        degradation_time = None

        degradation_position_pct = None

    # ========================================================
    # 24. 출력
    # ========================================================

    print("\n" + "=" * 40)

    print("Robust Moving Z-score 열화 탐지")

    print("=" * 40)

    print("열화 시작 시점:")

    print(degradation_time)

    print("전체 수명 위치(%):")

    print(degradation_position_pct)

    # ========================================================
    # 25. 데이터셋 저장 폴더
    # ========================================================

    dataset_dir = f"{SAVE_DIR}/" f"{dataset_name}"

    os.makedirs(dataset_dir, exist_ok=True)

    # ========================================================
    # 26. Baseline Median / MAD 저장
    # ========================================================

    baseline_stats_df.to_csv(
        f"{dataset_dir}/" "baseline_robust_statistics.csv", index=False
    )

    # ========================================================
    # 27. 열화 시작 원인 분석
    # ========================================================

    abnormal_features = []

    if degradation_idx is not None:

        print("\n" + "=" * 40)

        print("열화 시작 원인 분석")

        print("=" * 40)

        print("열화 시작 시점:")

        print(degradation_time)

        print("\n이상 채널 수:")

        print(df.loc[degradation_idx, "abnormal_channel_count"])

        # ----------------------------------------------------
        # Robust Z >= 3인 특징
        # ----------------------------------------------------

        for col in target_cols:

            flag_col = f"{col}_flag"

            z_col = f"{col}_zscore"

            if df.loc[degradation_idx, flag_col]:

                row = {
                    "dataset": dataset_name,
                    "degradation_start": degradation_time,
                    "feature": col,
                    "robust_zscore": df.loc[degradation_idx, z_col],
                }

                abnormal_features.append(row)

                cause_results.append(row)

        abnormal_feature_df = pd.DataFrame(abnormal_features)

        print("\n열화 시작 시점 이상 특징")

        if not abnormal_feature_df.empty:

            print(abnormal_feature_df[["feature", "robust_zscore"]])

        else:

            print("없음")

        # ----------------------------------------------------
        # 이상 채널 확인
        # ----------------------------------------------------

        abnormal_channels = []

        for channel in channels:

            channel_col = f"channel_{channel}_abnormal"

            if df.loc[degradation_idx, channel_col]:

                abnormal_channels.append(channel)

        print("\n이상 채널")

        print(abnormal_channels)

        # ----------------------------------------------------
        # 데이터셋별 원인 특징 저장
        # ----------------------------------------------------

        if not abnormal_feature_df.empty:

            abnormal_feature_df.to_csv(
                f"{dataset_dir}/" "degradation_start_features.csv", index=False
            )

    # ========================================================
    # 28. 상세 데이터 저장
    #
    # 06번은 이 파일의 *_zscore 컬럼을 그대로 사용
    # ========================================================

    df.to_csv(f"{dataset_dir}/" f"{dataset_name}_moving_zscore_detail.csv", index=False)

    # ========================================================
    # 29. 이상 채널 개수 그래프
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(df["start_time"], df["abnormal_channel_count"], label="이상 채널 수")

    plt.axhline(y=MIN_CHANNELS, linestyle="--", label="열화 판정 기준")

    if degradation_time is not None:

        plt.axvline(x=degradation_time, linestyle="--", label="열화 시작")

    plt.title(f"{dataset_name} Robust Moving Z-score 열화 탐지")

    plt.xlabel("Time")

    plt.ylabel("Abnormal Channel Count")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{dataset_dir}/" "moving_zscore_detection.png", dpi=300, bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ========================================================
    # 30. 필수 특징별 Robust Z-score 그래프
    # ========================================================

    for col in target_cols:

        z_col = f"{col}_zscore"

        plt.figure(figsize=(14, 4))

        plt.plot(df["start_time"], df[z_col], label=col)

        plt.axhline(y=Z_THRESHOLD, linestyle="--", label="Robust Z = 3")

        if degradation_time is not None:

            plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")

        plt.title(f"{dataset_name} - {col} Robust Moving Z-score")

        plt.xlabel("Time")

        plt.ylabel("Robust Z-score")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"{dataset_dir}/" f"{col}_moving_zscore.png", dpi=300, bbox_inches="tight"
        )

        plt.close()

    # ========================================================
    # 31. 요약 저장
    # ========================================================

    summary_results.append(
        {
            "dataset": dataset_name,
            "baseline_ratio": BASELINE_RATIO,
            "zscore_method": "median_MAD",
            "z_threshold": Z_THRESHOLD,
            "rolling_window": ROLLING_WINDOW,
            "min_channels": MIN_CHANNELS,
            "consecutive_count": CONSECUTIVE_COUNT,
            "degradation_start": degradation_time,
            "degradation_position_pct": degradation_position_pct,
        }
    )


# ============================================================
# 32. 전체 Moving Robust Z-score 결과
# ============================================================

summary_df = pd.DataFrame(summary_results)


summary_df.to_csv(f"{SAVE_DIR}/" "moving_zscore_summary.csv", index=False)


print("\n" + "=" * 70)

print("최종 Robust Moving Z-score 결과")

print("=" * 70)


print(summary_df)


# ============================================================
# 33. 전체 열화 시작 원인 저장
# ============================================================

cause_df = pd.DataFrame(cause_results)


cause_df.to_csv(f"{SAVE_DIR}/" "degradation_start_causes.csv", index=False)


print("\n" + "=" * 70)

print("전체 열화 시작 원인 특징")

print("=" * 70)


print(cause_df)


# ============================================================
# 34. 완료
# ============================================================

print("\n" + "=" * 70)

print("05 Robust Moving Z-score 분석 완료")

print("=" * 70)
