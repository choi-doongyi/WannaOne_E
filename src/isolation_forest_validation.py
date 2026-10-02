import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import (
    BASELINE_RATIO, IF_ANOMALY_PERCENTILE, IF_CONSECUTIVE_COUNT,
    IF_N_ESTIMATORS, IF_RANDOM_STATE, IF_ROLLING_ANOMALY_RATIO,
    IF_ROLLING_WINDOW, ISOLATION_FOREST_DIR, ISOLATION_FOREST_SUMMARY,
    MODEL_READY_DATASETS, MOVING_ZSCORE_SUMMARY,
)

def find_consecutive_start(series, consecutive_count):
    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()
    detected = rolling_count >= consecutive_count
    idx = detected[detected].index
    if len(idx) == 0:
        return None
    return idx[0] - consecutive_count + 1

def run():
    setup_plot_font()
    ensure_dir(ISOLATION_FOREST_DIR)
    zscore_summary = pd.read_csv(MOVING_ZSCORE_SUMMARY, parse_dates=["degradation_start"])
    summary_results = []

    for dataset_name, file_path in MODEL_READY_DATASETS.items():
        print("\n" + "=" * 70)
        print(dataset_name)
        print("=" * 70)
        df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
        df = df.sort_values("start_time").reset_index(drop=True)
        print("\n데이터 Shape:")
        print(df.shape)
        feature_cols = [col for col in df.columns if (col.endswith("_rms") or col.endswith("_kurtosis") or col.endswith("_max") or col.endswith("_crest_factor"))]
        print("\nIsolation Forest Feature 개수:")
        print(len(feature_cols))
        X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        baseline_end = int(len(df) * BASELINE_RATIO)
        X_baseline = X.iloc[:baseline_end].copy()
        baseline_end_time = df.loc[baseline_end - 1, "start_time"]
        print("\nBaseline 개수:", len(X_baseline))
        print("Baseline 종료:", baseline_end_time)
        model = IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination="auto", random_state=IF_RANDOM_STATE, n_jobs=-1)
        model.fit(X_baseline)
        df["if_anomaly_score"] = -model.score_samples(X)
        baseline_scores = df.loc[: baseline_end - 1, "if_anomaly_score"]
        anomaly_threshold = baseline_scores.quantile(IF_ANOMALY_PERCENTILE / 100)
        print("\nIF 이상 점수 Threshold:", anomaly_threshold)
        df["if_anomaly"] = df["if_anomaly_score"] >= anomaly_threshold
        df["if_rolling_anomaly_ratio"] = df["if_anomaly"].astype(int).rolling(window=IF_ROLLING_WINDOW, min_periods=IF_ROLLING_WINDOW).mean()
        df["if_degradation_candidate"] = df["if_rolling_anomaly_ratio"] >= IF_ROLLING_ANOMALY_RATIO
        df.loc[: baseline_end - 1, "if_degradation_candidate"] = False
        if_idx = find_consecutive_start(df["if_degradation_candidate"], IF_CONSECUTIVE_COUNT)
        if if_idx is not None:
            if_start_time = df.loc[if_idx, "start_time"]
            if_position_pct = if_idx / len(df) * 100
        else:
            if_start_time = None
            if_position_pct = None
        zscore_row = zscore_summary[zscore_summary["dataset"] == dataset_name]
        if not zscore_row.empty:
            zscore_start_time = zscore_row["degradation_start"].iloc[0]
            zscore_position_pct = zscore_row["degradation_position_pct"].iloc[0]
        else:
            zscore_start_time = zscore_position_pct = None
        if pd.notna(zscore_start_time) and if_start_time is not None:
            detection_time_difference = if_start_time - zscore_start_time
            detection_position_difference = if_position_pct - zscore_position_pct
        else:
            detection_time_difference = detection_position_difference = None
        print("\n" + "=" * 50)
        print("Isolation Forest 검증 결과")
        print("=" * 50)
        print("Moving Z-score 열화 시작:", zscore_start_time)
        print("Moving Z-score 수명 위치(%):", zscore_position_pct)
        print("Isolation Forest 이상 시작:", if_start_time)
        print("Isolation Forest 수명 위치(%):", if_position_pct)
        print("IF - Z-score 시간 차이:", detection_time_difference)
        print("IF - Z-score 위치 차이(%p):", detection_position_difference)
        dataset_dir = ensure_dir(ISOLATION_FOREST_DIR / dataset_name)
        df.to_csv(dataset_dir / f"{dataset_name}_isolation_forest_detail.csv", index=False)

        plt.figure(figsize=(14, 5))
        plt.plot(df["start_time"], df["if_anomaly_score"], label="Isolation Forest Anomaly Score")
        plt.axhline(y=anomaly_threshold, linestyle="--", label="Baseline 99% Threshold")
        if pd.notna(zscore_start_time):
            plt.axvline(x=zscore_start_time, linestyle=":", label="Moving Z-score 열화 시작")
        if if_start_time is not None:
            plt.axvline(x=if_start_time, linestyle="--", label="Isolation Forest 이상 시작")
        plt.title(f"{dataset_name} Isolation Forest Anomaly Score")
        plt.xlabel("Time")
        plt.ylabel("Anomaly Score")
        plt.legend()
        finish_plot(dataset_dir / "isolation_forest_anomaly_score.png")

        plt.figure(figsize=(14, 5))
        plt.plot(df["start_time"], df["if_rolling_anomaly_ratio"], label="Rolling Anomaly Ratio")
        plt.axhline(y=IF_ROLLING_ANOMALY_RATIO, linestyle="--", label="판정 기준")
        if pd.notna(zscore_start_time):
            plt.axvline(x=zscore_start_time, linestyle=":", label="Moving Z-score 열화 시작")
        if if_start_time is not None:
            plt.axvline(x=if_start_time, linestyle="--", label="Isolation Forest 이상 시작")
        plt.title(f"{dataset_name} Isolation Forest 이상 발생 비율")
        plt.xlabel("Time")
        plt.ylabel("Rolling Anomaly Ratio")
        plt.legend()
        finish_plot(dataset_dir / "isolation_forest_rolling_ratio.png")

        summary_results.append({
            "dataset": dataset_name,
            "baseline_ratio": BASELINE_RATIO,
            "if_threshold_percentile": IF_ANOMALY_PERCENTILE,
            "rolling_window": IF_ROLLING_WINDOW,
            "rolling_anomaly_ratio": IF_ROLLING_ANOMALY_RATIO,
            "consecutive_count": IF_CONSECUTIVE_COUNT,
            "zscore_start": zscore_start_time,
            "zscore_position_pct": zscore_position_pct,
            "if_start": if_start_time,
            "if_position_pct": if_position_pct,
            "if_minus_zscore_time": detection_time_difference,
            "if_minus_zscore_position_pct": detection_position_difference,
        })

    summary_result_df = pd.DataFrame(summary_results)
    summary_result_df.to_csv(ISOLATION_FOREST_SUMMARY, index=False)
    print("\n" + "=" * 70)
    print("최종 Isolation Forest 검증 결과")
    print("=" * 70)
    print(summary_result_df)

if __name__ == "__main__":
    run()
