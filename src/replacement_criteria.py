import re
import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import (
    MIN_FEATURE_TYPES, MIN_SEVERE_CHANNELS, MOVING_ZSCORE_DETAIL_PATHS,
    MOVING_ZSCORE_SUMMARY, REPLACEMENT_CONSECUTIVE_COUNT, REPLACEMENT_DIR,
    REPLACEMENT_FEATURE_TYPES, REPLACEMENT_SUMMARY, SEVERE_Z_THRESHOLD,
)

def get_channel_number(column_name):
    m = re.search(r"channel_(\d+)_", column_name)
    return int(m.group(1)) if m else None

def get_feature_type(column_name):
    if "_crest_factor_" in column_name: return "crest_factor"
    if "_kurtosis_" in column_name: return "kurtosis"
    if "_rms_" in column_name: return "rms"
    if "_max_" in column_name: return "max"
    return None

def find_consecutive_start(series, consecutive_count):
    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()
    detected = rolling_count[rolling_count >= consecutive_count]
    if len(detected) == 0:
        return None
    return detected.index[0] - consecutive_count + 1

def run():
    setup_plot_font()
    ensure_dir(REPLACEMENT_DIR)
    summary_df = pd.read_csv(MOVING_ZSCORE_SUMMARY, parse_dates=["degradation_start"])
    replacement_results, replacement_cause_results = [], []

    for dataset_name, detail_path in MOVING_ZSCORE_DETAIL_PATHS.items():
        print("\n" + "=" * 70)
        print(dataset_name)
        print("=" * 70)
        df = pd.read_csv(detail_path, parse_dates=["start_time", "end_time"])
        df = df.sort_values("start_time").reset_index(drop=True)
        degradation_row = summary_df[summary_df["dataset"] == dataset_name]
        if degradation_row.empty:
            print("열화 시작 결과 없음")
            continue
        degradation_time = degradation_row["degradation_start"].iloc[0]
        if pd.isna(degradation_time):
            print("열화 시작 시점 없음")
            continue
        print("\n열화 시작:")
        print(degradation_time)

        zscore_cols = [col for col in df.columns if (col.endswith("_rms_zscore") or col.endswith("_kurtosis_zscore") or col.endswith("_max_zscore") or col.endswith("_crest_factor_zscore"))]
        print("\n사용 Z-score 특징 개수:")
        print(len(zscore_cols))
        severe_feature_df = pd.DataFrame({col: df[col] >= SEVERE_Z_THRESHOLD for col in zscore_cols}, index=df.index)
        channels = sorted({get_channel_number(col) for col in zscore_cols if get_channel_number(col) is not None})
        print("\n채널:")
        print(channels)
        channel_severe_df = pd.DataFrame({
            f"channel_{channel}_severe": severe_feature_df[[col for col in zscore_cols if col.startswith(f"channel_{channel}_")]].any(axis=1)
            for channel in channels
        }, index=df.index)
        feature_type_df = pd.DataFrame(index=df.index)
        for feature_type in REPLACEMENT_FEATURE_TYPES:
            type_cols = [col for col in zscore_cols if get_feature_type(col) == feature_type]
            feature_type_df[f"{feature_type}_severe"] = severe_feature_df[type_cols].any(axis=1) if type_cols else False
        severe_channel_count = channel_severe_df.sum(axis=1)
        severe_feature_type_count = feature_type_df.sum(axis=1)
        analysis_cols = pd.concat([
            channel_severe_df,
            feature_type_df,
            severe_channel_count.rename("severe_channel_count"),
            severe_feature_type_count.rename("severe_feature_type_count"),
        ], axis=1)
        df = pd.concat([df, analysis_cols], axis=1)
        df["replacement_candidate"] = (df["severe_channel_count"] >= MIN_SEVERE_CHANNELS) & (df["severe_feature_type_count"] >= MIN_FEATURE_TYPES)
        df.loc[df["start_time"] < degradation_time, "replacement_candidate"] = False
        replacement_idx = find_consecutive_start(df["replacement_candidate"], REPLACEMENT_CONSECUTIVE_COUNT)
        dataset_end_time = df["end_time"].iloc[-1]
        if replacement_idx is not None:
            replacement_time = df.loc[replacement_idx, "start_time"]
            replacement_position_pct = replacement_idx / len(df) * 100
            degradation_to_replacement = replacement_time - degradation_time
            replacement_to_dataset_end = dataset_end_time - replacement_time
        else:
            replacement_time = replacement_position_pct = degradation_to_replacement = replacement_to_dataset_end = None

        print("\n==============================")
        print("교체 권고 결과")
        print("==============================")
        print("열화 시작:", degradation_time)
        print("교체 권고:", replacement_time)
        print("교체 권고 수명 위치(%):", replacement_position_pct)
        print("열화 시작 → 교체 권고:", degradation_to_replacement)
        print("교체 권고 → 데이터 종료:", replacement_to_dataset_end)

        abnormal_features = []
        if replacement_idx is not None:
            for col in zscore_cols:
                z_value = df.loc[replacement_idx, col]
                if z_value >= SEVERE_Z_THRESHOLD:
                    row = {
                        "dataset": dataset_name,
                        "replacement_time": replacement_time,
                        "channel": get_channel_number(col),
                        "feature_type": get_feature_type(col),
                        "feature": col.replace("_zscore", ""),
                        "zscore": z_value,
                    }
                    abnormal_features.append(row)
                    replacement_cause_results.append(row)
        abnormal_feature_df = pd.DataFrame(abnormal_features)
        print("\n교체 권고 원인 특징")
        print(abnormal_feature_df[["channel", "feature_type", "feature", "zscore"]] if not abnormal_feature_df.empty else "없음")

        dataset_dir = ensure_dir(REPLACEMENT_DIR / dataset_name)
        df.to_csv(dataset_dir / f"{dataset_name}_replacement_detail.csv", index=False)
        if not abnormal_feature_df.empty:
            abnormal_feature_df.to_csv(dataset_dir / "replacement_features.csv", index=False)

        plt.figure(figsize=(14, 5))
        plt.plot(df["start_time"], df["severe_channel_count"], label="심각 이상 채널 수")
        plt.axhline(y=MIN_SEVERE_CHANNELS, linestyle="--", label="채널 기준")
        plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")
        if replacement_time is not None:
            plt.axvline(x=replacement_time, linestyle="--", label="교체 권고")
        plt.title(f"{dataset_name} 심각 이상 채널 수")
        plt.xlabel("Time")
        plt.ylabel("Severe Channel Count")
        plt.legend()
        finish_plot(dataset_dir / "severe_channel_count.png")

        plt.figure(figsize=(14, 5))
        plt.plot(df["start_time"], df["severe_feature_type_count"], label="심각 이상 특징 종류 수")
        plt.axhline(y=MIN_FEATURE_TYPES, linestyle="--", label="특징 종류 기준")
        plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")
        if replacement_time is not None:
            plt.axvline(x=replacement_time, linestyle="--", label="교체 권고")
        plt.title(f"{dataset_name} 심각 이상 특징 종류")
        plt.xlabel("Time")
        plt.ylabel("Severe Feature Type Count")
        plt.legend()
        finish_plot(dataset_dir / "severe_feature_type_count.png")

        replacement_results.append({
            "dataset": dataset_name,
            "degradation_start": degradation_time,
            "severe_z_threshold": SEVERE_Z_THRESHOLD,
            "min_severe_channels": MIN_SEVERE_CHANNELS,
            "min_feature_types": MIN_FEATURE_TYPES,
            "consecutive_count": REPLACEMENT_CONSECUTIVE_COUNT,
            "replacement_time": replacement_time,
            "replacement_position_pct": replacement_position_pct,
            "degradation_to_replacement": degradation_to_replacement,
            "dataset_end_time": dataset_end_time,
            "replacement_to_dataset_end": replacement_to_dataset_end,
        })

    result_df = pd.DataFrame(replacement_results)
    result_df.to_csv(REPLACEMENT_SUMMARY, index=False)
    cause_df = pd.DataFrame(replacement_cause_results)
    cause_df.to_csv(REPLACEMENT_DIR / "replacement_causes.csv", index=False)
    print("\n" + "=" * 70)
    print("최종 교체 권고 결과")
    print("=" * 70)
    print(result_df)

if __name__ == "__main__":
    run()
