import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import (
    BASELINE_RATIO,
    MIN_CHANNELS,
    MODEL_READY_DATASETS,
    MOVING_ZSCORE_DIR,
    MOVING_ZSCORE_SUMMARY,
    ROBUST_SCALE,
    Z_THRESHOLD,
    ZSCORE_CONSECUTIVE_COUNT,
    ZSCORE_ROLLING_WINDOW,
)


def get_channel_number(column_name):
    m = re.search(r"channel_(\d+)_", column_name)
    return int(m.group(1)) if m else None


def find_consecutive_start(series, consecutive_count):
    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()
    detected = rolling_count[rolling_count >= consecutive_count]
    if len(detected) == 0:
        return None
    end_idx = detected.index[0]
    return end_idx - consecutive_count + 1


def calculate_mad(series):
    median_value = series.median()
    mad_value = (series - median_value).abs().median()
    return median_value, mad_value


def run():
    setup_plot_font()
    ensure_dir(MOVING_ZSCORE_DIR)
    summary_results = []
    cause_results = []

    for dataset_name, file_path in MODEL_READY_DATASETS.items():
        print("\n" + "=" * 70)
        print(dataset_name)
        print("=" * 70)
        df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
        df = df.sort_values("start_time").reset_index(drop=True)
        print("\n데이터 Shape:")
        print(df.shape)
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
        baseline_end = int(len(df) * BASELINE_RATIO)
        baseline = df.iloc[:baseline_end].copy()
        print("\nBaseline 데이터 개수:")
        print(len(baseline))
        print("\nBaseline 종료 시점:")
        print(baseline["start_time"].iloc[-1])

        moving_data, zscore_data, flag_data, baseline_stats = {}, {}, {}, []
        for col in target_cols:
            moving_series = (
                df[col].rolling(window=ZSCORE_ROLLING_WINDOW, min_periods=1).mean()
            )
            moving_data[f"{col}_moving"] = moving_series
            baseline_moving = (
                baseline[col]
                .rolling(window=ZSCORE_ROLLING_WINDOW, min_periods=1)
                .mean()
            )
            baseline_median, baseline_mad = calculate_mad(baseline_moving)
            if pd.isna(baseline_mad) or baseline_mad == 0:
                q1, q3 = baseline_moving.quantile(0.25), baseline_moving.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    robust_z = pd.Series(0.0, index=df.index)
                else:
                    robust_z = (moving_series - baseline_median) / (iqr / 1.349)
            else:
                robust_z = (
                    ROBUST_SCALE * (moving_series - baseline_median) / baseline_mad
                )
            zscore_data[f"{col}_zscore"] = robust_z
            flag_data[f"{col}_flag"] = robust_z >= Z_THRESHOLD
            baseline_stats.append(
                {
                    "feature": col,
                    "baseline_median": baseline_median,
                    "baseline_mad": baseline_mad,
                }
            )

        moving_df = pd.DataFrame(moving_data)
        zscore_df = pd.DataFrame(zscore_data)
        flag_df = pd.DataFrame(flag_data)
        df = pd.concat([df, moving_df, zscore_df, flag_df], axis=1)
        baseline_stats_df = pd.DataFrame(baseline_stats)

        channels = sorted(
            {
                get_channel_number(col)
                for col in target_cols
                if get_channel_number(col) is not None
            }
        )
        print("\n채널:")
        print(channels)
        channel_flag_data = {}
        for channel in channels:
            current_flags = [
                col for col in flag_df.columns if col.startswith(f"channel_{channel}_")
            ]
            channel_flag_data[f"channel_{channel}_abnormal"] = flag_df[
                current_flags
            ].any(axis=1)
        channel_flag_df = pd.DataFrame(channel_flag_data)
        df = pd.concat([df, channel_flag_df], axis=1)
        df["abnormal_channel_count"] = channel_flag_df.sum(axis=1)
        df["degradation_candidate"] = df["abnormal_channel_count"] >= MIN_CHANNELS
        df.loc[: baseline_end - 1, "degradation_candidate"] = False
        degradation_idx = find_consecutive_start(
            df["degradation_candidate"], ZSCORE_CONSECUTIVE_COUNT
        )

        if degradation_idx is not None:
            degradation_time = df.loc[degradation_idx, "start_time"]
            degradation_position_pct = degradation_idx / len(df) * 100
        else:
            degradation_time = None
            degradation_position_pct = None

        print("\n" + "=" * 40)
        print("Robust Moving Z-score 열화 탐지")
        print("=" * 40)
        print("열화 시작 시점:")
        print(degradation_time)
        print("전체 수명 위치(%):")
        print(degradation_position_pct)

        dataset_dir = ensure_dir(MOVING_ZSCORE_DIR / dataset_name)
        baseline_stats_df.to_csv(
            dataset_dir / "baseline_robust_statistics.csv", index=False
        )

        abnormal_features = []
        if degradation_idx is not None:
            for col in target_cols:
                if df.loc[degradation_idx, f"{col}_flag"]:
                    row = {
                        "dataset": dataset_name,
                        "degradation_start": degradation_time,
                        "feature": col,
                        "robust_zscore": df.loc[degradation_idx, f"{col}_zscore"],
                    }
                    abnormal_features.append(row)
                    cause_results.append(row)
            abnormal_feature_df = pd.DataFrame(abnormal_features)
            print("\n열화 시작 시점 이상 특징")
            print(
                abnormal_feature_df[["feature", "robust_zscore"]]
                if not abnormal_feature_df.empty
                else "없음"
            )
            if not abnormal_feature_df.empty:
                abnormal_feature_df.to_csv(
                    dataset_dir / "degradation_start_features.csv", index=False
                )

        df.to_csv(dataset_dir / f"{dataset_name}_moving_zscore_detail.csv", index=False)

        plt.figure(figsize=(14, 5))
        plt.plot(df["start_time"], df["abnormal_channel_count"], label="이상 채널 수")
        plt.axhline(y=MIN_CHANNELS, linestyle="--", label="열화 판정 기준")
        if degradation_time is not None:
            plt.axvline(x=degradation_time, linestyle="--", label="열화 시작")
        plt.title(f"{dataset_name} Robust Moving Z-score 열화 탐지")
        plt.xlabel("Time")
        plt.ylabel("Abnormal Channel Count")
        plt.legend()
        finish_plot(dataset_dir / "moving_zscore_detection.png")

        for col in target_cols:
            z_col = f"{col}_zscore"
            plt.figure(figsize=(14, 4))
            plt.plot(df["start_time"], df[z_col], label=col)
            plt.axhline(
                y=Z_THRESHOLD, linestyle="--", label=f"Robust Z = {Z_THRESHOLD:g}"
            )
            if degradation_time is not None:
                plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")
            plt.title(f"{dataset_name} - {col} Robust Moving Z-score")
            plt.xlabel("Time")
            plt.ylabel("Robust Z-score")
            plt.legend()
            finish_plot(dataset_dir / f"{col}_moving_zscore.png")

        summary_results.append(
            {
                "dataset": dataset_name,
                "baseline_ratio": BASELINE_RATIO,
                "zscore_method": "median_MAD",
                "z_threshold": Z_THRESHOLD,
                "rolling_window": ZSCORE_ROLLING_WINDOW,
                "min_channels": MIN_CHANNELS,
                "consecutive_count": ZSCORE_CONSECUTIVE_COUNT,
                "degradation_start": degradation_time,
                "degradation_position_pct": degradation_position_pct,
            }
        )

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(MOVING_ZSCORE_SUMMARY, index=False)
    cause_df = pd.DataFrame(cause_results)
    cause_df.to_csv(MOVING_ZSCORE_DIR / "degradation_start_causes.csv", index=False)
    print("\n" + "=" * 70)
    print("최종 Robust Moving Z-score 결과")
    print("=" * 70)
    print(summary_df)


if __name__ == "__main__":
    run()
