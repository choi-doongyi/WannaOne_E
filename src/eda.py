import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import EDA_DATASET, EDA_DIR, PROCESSED_DATASETS

FEATURE_KEYWORDS = ["rms", "kurtosis", "crest_factor", "peak_to_peak", "spectral_energy", "dominant_freq", "spectral_centroid", "spectral_entropy"]

def run():
    setup_plot_font()
    df = pd.read_csv(PROCESSED_DATASETS[EDA_DATASET], parse_dates=["start_time", "end_time"])
    df = df.sort_values("start_time").reset_index(drop=True)

    time_trend_dir = ensure_dir(EDA_DIR / "time_trend")
    for keyword in FEATURE_KEYWORDS:
        cols = [col for col in df.columns if keyword in col.lower()]
        if not cols:
            continue
        plt.figure(figsize=(14, 5))
        for col in cols:
            plt.plot(df["start_time"], df[col], label=col, alpha=0.8)
        plt.title(f"{keyword} 시간에 따른 변화")
        plt.xlabel("Time")
        plt.ylabel(keyword)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        finish_plot(time_trend_dir / f"{keyword}_time_trend.png")

    feature_cols = [col for col in df.select_dtypes(include="number").columns if col not in ["segment", "n_samples"]]
    skewness = df[feature_cols].skew().sort_values(ascending=False)
    print("\n==============================")
    print("왜도 확인")
    print("==============================")
    print(skewness)

    correlation_dir = ensure_dir(EDA_DIR / "correlation")
    corr = df[feature_cols].corr()
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if abs(value) >= 0.9:
                high_corr_pairs.append({"feature_1": corr.columns[i], "feature_2": corr.columns[j], "correlation": value})
    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df["abs_correlation"] = high_corr_df["correlation"].abs()
        high_corr_df = high_corr_df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    print("\n==============================")
    print("|상관계수| >= 0.9 특징쌍")
    print("==============================")
    print(high_corr_df.head(30))
    high_corr_df.to_csv(correlation_dir / "high_correlation_pairs.csv", index=False)

    n = len(df)
    early_end = int(n * 0.3)
    late_start = int(n * 0.7)
    df["stage"] = "중기"
    df.loc[: early_end - 1, "stage"] = "초기"
    df.loc[late_start:, "stage"] = "후기"
    degradation_dir = ensure_dir(EDA_DIR / "degradation_stage")
    stage_mean = df.groupby("stage")[feature_cols].mean().reindex(["초기", "중기", "후기"])
    change_df = pd.DataFrame({"early_mean": stage_mean.loc["초기"], "middle_mean": stage_mean.loc["중기"], "late_mean": stage_mean.loc["후기"]})
    change_df["late_vs_early_ratio"] = change_df["late_mean"] / change_df["early_mean"]
    change_df["difference"] = change_df["late_mean"] - change_df["early_mean"]
    change_df = change_df.sort_values("late_vs_early_ratio", ascending=False)
    change_df.to_csv(degradation_dir / "stage_feature_change.csv")
    print(change_df.head(20))

if __name__ == "__main__":
    run()
