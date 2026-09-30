# ============================================================
# 07_Isolation_Forest_Validation.py
#
# Isolation Forest를 이용한
# Moving Z-score 열화 시작 시점 보조 검증
#
# 목적:
# - 초기 정상구간만 이용해 Isolation Forest 학습
# - 전체 기간의 이상 점수 계산
# - 이상 발생이 지속적으로 증가하는 최초 시점 탐지
# - 05 Moving Z-score 결과와 비교
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

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
# 3. Moving Z-score 결과
# ============================================================

ZSCORE_SUMMARY_PATH = "results/moving_zscore/" "moving_zscore_summary.csv"


# ============================================================
# 4. 저장 폴더
# ============================================================

SAVE_DIR = "results/isolation_forest_validation"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 5. 분석 설정
# ============================================================

# 05번과 동일하게 초기 30%를 정상구간으로 사용
BASELINE_RATIO = 0.30


# Isolation Forest 설정
N_ESTIMATORS = 300

RANDOM_STATE = 42


# ============================================================
# 이상 점수 기준
#
# 초기 정상구간 anomaly score 중
# 상위 1% 수준을 이상 기준으로 설정
#
# 즉 baseline의 99 percentile
# ============================================================

ANOMALY_PERCENTILE = 99


# ============================================================
# Rolling 조건
#
# 최근 5개의 측정 segment 중
# 60% 이상이 이상이면 이상 구간 후보
# ============================================================

ROLLING_WINDOW = 5

ROLLING_ANOMALY_RATIO = 0.60


# ============================================================
# 해당 상태가 3 segment 연속 발생하면
# IF 이상 시작점으로 판단
# ============================================================

CONSECUTIVE_COUNT = 3


# ============================================================
# 6. 연속 True 시작점 탐색 함수
# ============================================================


def find_consecutive_start(series, consecutive_count):

    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()

    detected = rolling_count >= consecutive_count

    detected_index = detected[detected].index

    if len(detected_index) == 0:

        return None

    end_idx = detected_index[0]

    start_idx = end_idx - consecutive_count + 1

    return start_idx


# ============================================================
# 7. Moving Z-score 결과 불러오기
# ============================================================

zscore_summary = pd.read_csv(ZSCORE_SUMMARY_PATH, parse_dates=["degradation_start"])


# ============================================================
# 8. 전체 결과 저장용
# ============================================================

summary_results = []


# ============================================================
# 9. 데이터셋별 실행
# ============================================================

for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 70)

    print(dataset_name)

    print("=" * 70)

    # ========================================================
    # 10. 데이터 불러오기
    # ========================================================

    df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])

    df = df.sort_values("start_time").reset_index(drop=True)

    print("\n데이터 Shape:")

    print(df.shape)

    # ========================================================
    # 11. 필수 특징만 선택
    #
    # RMS
    # Kurtosis
    # Max
    # Crest Factor
    #
    # FFT 특징은 여기서 사용하지 않음
    # ========================================================

    feature_cols = [
        col
        for col in df.columns
        if (
            col.endswith("_rms")
            or col.endswith("_kurtosis")
            or col.endswith("_max")
            or col.endswith("_crest_factor")
        )
    ]

    print("\nIsolation Forest Feature 개수:")

    print(len(feature_cols))

    # ========================================================
    # 12. 결측 / inf 최종 확인
    # ========================================================

    X = df[feature_cols].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # 혹시 남아있다면 중앙값
    for col in feature_cols:

        if X[col].isna().any():

            X[col] = X[col].fillna(X[col].median())

    # ========================================================
    # 13. 초기 정상구간
    # ========================================================

    baseline_end = int(len(df) * BASELINE_RATIO)

    X_baseline = X.iloc[:baseline_end].copy()

    baseline_end_time = df.loc[baseline_end - 1, "start_time"]

    print("\nBaseline 개수:")

    print(len(X_baseline))

    print("Baseline 종료:")

    print(baseline_end_time)

    # ========================================================
    # 14. Isolation Forest 모델
    #
    # 초기 정상구간만 fit
    # ========================================================

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_baseline)

    # ========================================================
    # 15. 전체 데이터 이상 점수
    #
    # score_samples:
    # 값이 낮을수록 이상
    #
    # 사람이 보기 편하게 음수를 붙여
    # 값이 클수록 이상하도록 변환
    # ========================================================

    df["if_anomaly_score"] = -model.score_samples(X)

    # ========================================================
    # 16. 정상구간 이상 점수 기준선
    #
    # baseline anomaly score의
    # 99 percentile 사용
    # ========================================================

    baseline_scores = df.loc[: baseline_end - 1, "if_anomaly_score"]

    anomaly_threshold = baseline_scores.quantile(ANOMALY_PERCENTILE / 100)

    print("\nIF 이상 점수 Threshold:")

    print(anomaly_threshold)

    # ========================================================
    # 17. 각 segment 이상 여부
    # ========================================================

    df["if_anomaly"] = df["if_anomaly_score"] >= anomaly_threshold

    # ========================================================
    # 18. Rolling 이상 비율
    #
    # 최근 5개 segment 중
    # 이상으로 판정된 비율
    # ========================================================

    df["if_rolling_anomaly_ratio"] = (
        df["if_anomaly"]
        .astype(int)
        .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .mean()
    )

    # ========================================================
    # 19. IF 이상 구간 후보
    #
    # 최근 5개 중 60% 이상 이상
    # ========================================================

    df["if_degradation_candidate"] = (
        df["if_rolling_anomaly_ratio"] >= ROLLING_ANOMALY_RATIO
    )

    # ========================================================
    # 20. Baseline 내부 탐지 금지
    # ========================================================

    df.loc[: baseline_end - 1, "if_degradation_candidate"] = False

    # ========================================================
    # 21. 3 segment 연속 조건
    # ========================================================

    if_idx = find_consecutive_start(df["if_degradation_candidate"], CONSECUTIVE_COUNT)

    # ========================================================
    # 22. IF 탐지 결과
    # ========================================================

    if if_idx is not None:

        if_start_time = df.loc[if_idx, "start_time"]

        if_position_pct = if_idx / len(df) * 100

    else:

        if_start_time = None

        if_position_pct = None

    # ========================================================
    # 23. 05 Moving Z-score 열화 시작
    # ========================================================

    zscore_row = zscore_summary[zscore_summary["dataset"] == dataset_name]

    if not zscore_row.empty:

        zscore_start_time = zscore_row["degradation_start"].iloc[0]

        zscore_position_pct = zscore_row["degradation_position_pct"].iloc[0]

    else:

        zscore_start_time = None

        zscore_position_pct = None

    # ========================================================
    # 24. 두 탐지 방법 시간 차이
    # ========================================================

    if pd.notna(zscore_start_time) and if_start_time is not None:

        detection_time_difference = if_start_time - zscore_start_time

        detection_position_difference = if_position_pct - zscore_position_pct

    else:

        detection_time_difference = None

        detection_position_difference = None

    # ========================================================
    # 25. 출력
    # ========================================================

    print("\n" + "=" * 50)

    print("Isolation Forest 검증 결과")

    print("=" * 50)

    print("Moving Z-score 열화 시작:")

    print(zscore_start_time)

    print("Moving Z-score 수명 위치(%):")

    print(zscore_position_pct)

    print("\nIsolation Forest 이상 시작:")

    print(if_start_time)

    print("Isolation Forest 수명 위치(%):")

    print(if_position_pct)

    print("\nIF - Z-score 시간 차이:")

    print(detection_time_difference)

    print("IF - Z-score 위치 차이(%p):")

    print(detection_position_difference)

    # ========================================================
    # 26. 데이터셋별 저장 폴더
    # ========================================================

    dataset_dir = f"{SAVE_DIR}/" f"{dataset_name}"

    os.makedirs(dataset_dir, exist_ok=True)

    # ========================================================
    # 27. 상세 결과 저장
    # ========================================================

    df.to_csv(
        f"{dataset_dir}/" f"{dataset_name}_isolation_forest_detail.csv", index=False
    )

    # ========================================================
    # 28. 시각화 1
    # Isolation Forest Anomaly Score
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["start_time"], df["if_anomaly_score"], label="Isolation Forest Anomaly Score"
    )

    plt.axhline(y=anomaly_threshold, linestyle="--", label="Baseline 99% Threshold")

    if pd.notna(zscore_start_time):

        plt.axvline(
            x=zscore_start_time, linestyle=":", label="Moving Z-score 열화 시작"
        )

    if if_start_time is not None:

        plt.axvline(x=if_start_time, linestyle="--", label="Isolation Forest 이상 시작")

    plt.title(f"{dataset_name} Isolation Forest Anomaly Score")

    plt.xlabel("Time")

    plt.ylabel("Anomaly Score")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{dataset_dir}/" "isolation_forest_anomaly_score.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ========================================================
    # 29. 시각화 2
    # Rolling anomaly ratio
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["start_time"], df["if_rolling_anomaly_ratio"], label="Rolling Anomaly Ratio"
    )

    plt.axhline(y=ROLLING_ANOMALY_RATIO, linestyle="--", label="판정 기준")

    if pd.notna(zscore_start_time):

        plt.axvline(
            x=zscore_start_time, linestyle=":", label="Moving Z-score 열화 시작"
        )

    if if_start_time is not None:

        plt.axvline(x=if_start_time, linestyle="--", label="Isolation Forest 이상 시작")

    plt.title(f"{dataset_name} Isolation Forest 이상 발생 비율")

    plt.xlabel("Time")

    plt.ylabel("Rolling Anomaly Ratio")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{dataset_dir}/" "isolation_forest_rolling_ratio.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ========================================================
    # 30. 요약 결과
    # ========================================================

    summary_results.append(
        {
            "dataset": dataset_name,
            "baseline_ratio": BASELINE_RATIO,
            "if_threshold_percentile": ANOMALY_PERCENTILE,
            "rolling_window": ROLLING_WINDOW,
            "rolling_anomaly_ratio": ROLLING_ANOMALY_RATIO,
            "consecutive_count": CONSECUTIVE_COUNT,
            "zscore_start": zscore_start_time,
            "zscore_position_pct": zscore_position_pct,
            "if_start": if_start_time,
            "if_position_pct": if_position_pct,
            "if_minus_zscore_time": detection_time_difference,
            "if_minus_zscore_position_pct": detection_position_difference,
        }
    )


# ============================================================
# 31. 전체 요약
# ============================================================

summary_result_df = pd.DataFrame(summary_results)


summary_result_df.to_csv(
    f"{SAVE_DIR}/" "isolation_forest_validation_summary.csv", index=False
)


# ============================================================
# 32. 출력
# ============================================================

print("\n" + "=" * 70)

print("최종 Isolation Forest 검증 결과")

print("=" * 70)


print(summary_result_df)


print("\n" + "=" * 70)

print("07 Isolation Forest 보조 검증 완료")

print("=" * 70)


# Moving Z-score는 열화 시작 탐지에 사용하고,
# Isolation Forest는 다변량 이상 상태가 본격적으로 형성되는 시점을 보조 분석하였다.
