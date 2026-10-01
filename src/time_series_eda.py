import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import EDA_DATASET, EDA_DIR, PROCESSED_DATASETS

FEATURE_KEYWORDS = ["rms", "kurtosis", "crest_factor", "peak_to_peak", "spectral_energy"]
ROLLING_WINDOW = 20
HI_ROLLING_WINDOW = 20

def run():
    setup_plot_font()
    df = pd.read_csv(PROCESSED_DATASETS[EDA_DATASET], parse_dates=["start_time", "end_time"])
    df = df.sort_values("start_time").reset_index(drop=True)
    print("데이터 크기:", df.shape)
    print("\n시간 범위")
    print(df["start_time"].min(), "~", df["start_time"].max())

    time_series_dir = ensure_dir(EDA_DIR / "time_series")
    cluster_dir = ensure_dir(EDA_DIR / "change_point_cluster")
    hi_dir = ensure_dir(EDA_DIR / "health_indicator")
    change_point_results = []

    for keyword in FEATURE_KEYWORDS:
        cols = [col for col in df.columns if keyword in col.lower()]
        for col in cols:
            rolling_mean = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
            rolling_diff = rolling_mean.diff().abs()
            if rolling_diff.notna().sum() == 0:
                continue
            change_idx = rolling_diff.idxmax()
            change_time = df.loc[change_idx, "start_time"]
            change_value = rolling_diff.loc[change_idx]
            change_point_results.append({"feature": col, "change_index": change_idx, "change_time": change_time, "change_value": change_value})
            print("\n==============================")
            print(col)
            print("==============================")
            print("가장 큰 변화 시점:", change_time)
            print("변화량:", change_value)
            plt.figure(figsize=(14, 5))
            plt.plot(df["start_time"], df[col], alpha=0.35, label="원본")
            plt.plot(df["start_time"], rolling_mean, linewidth=2, label=f"Rolling Mean ({ROLLING_WINDOW})")
            plt.axvline(x=change_time, linestyle="--", label="가장 큰 변화점")
            plt.title(f"{col} 시계열 변화점 탐색")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.legend()
            finish_plot(time_series_dir / f"{col}_change_point.png")

    change_point_df = pd.DataFrame(change_point_results)
    if change_point_df.empty:
        print("변화점 결과 없음")
        return
    change_point_df = change_point_df.sort_values("change_value", ascending=False).reset_index(drop=True)
    print("\n==============================")
    print("변화점 탐색 결과")
    print("==============================")
    print(change_point_df.head(20))
    change_point_df.to_csv(time_series_dir / "change_point_results.csv", index=False)

    change_point_df["change_time"] = pd.to_datetime(change_point_df["change_time"])
    change_point_sorted = change_point_df.sort_values("change_time").reset_index(drop=True)
    plt.figure(figsize=(14, 10))
    plt.scatter(change_point_sorted["change_time"], change_point_sorted["feature"])
    plt.title("특징별 변화점 발생 시점")
    plt.xlabel("Change Time")
    plt.ylabel("Feature")
    plt.xticks(rotation=45)
    finish_plot(cluster_dir / "change_point_timeline.png")

    change_point_sorted["change_date"] = change_point_sorted["change_time"].dt.floor("D")
    change_count = change_point_sorted.groupby("change_date").size().sort_values(ascending=False)
    print("\n==============================")
    print("날짜별 변화점 개수")
    print("==============================")
    print(change_count.head(20))
    change_count_plot = change_count.sort_index()
    plt.figure(figsize=(14, 5))
    plt.bar(change_count_plot.index, change_count_plot.values)
    plt.title("날짜별 변화점 발생 개수")
    plt.xlabel("Date")
    plt.ylabel("변화 특징 개수")
    plt.xticks(rotation=45)
    finish_plot(cluster_dir / "change_point_count_by_date.png")

    if len(change_count) > 0:
        peak_change_date = change_count.idxmax()
        peak_change_count = change_count.max()
        print("\n==============================")
        print("변화점 집중 시점")
        print("==============================")
        print("가장 많은 변화점이 발생한 날짜:", peak_change_date)
        print("해당 날짜 변화 특징 개수:", peak_change_count)
        peak_features = change_point_sorted[change_point_sorted["change_date"] == peak_change_date]
        print("\n해당 시점 변화 특징")
        print(peak_features[["feature", "change_time", "change_value"]])
        peak_features.to_csv(cluster_dir / "peak_change_features.csv", index=False)

    hi_keywords = ["rms", "kurtosis", "crest_factor", "peak_to_peak", "spectral_energy"]
    hi_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in hi_keywords)]
    print("\n==============================")
    print("Health Indicator 사용 특징")
    print("==============================")
    print(hi_cols)
    hi_data = df[hi_cols].copy()
    print("\nHI 특징 결측치 개수")
    print(hi_data.isna().sum().sum())
    hi_data = hi_data.fillna(hi_data.median())
    hi_scaled = StandardScaler().fit_transform(hi_data)
    df["health_indicator"] = hi_scaled.mean(axis=1)
    df["health_indicator_rolling"] = df["health_indicator"].rolling(window=HI_ROLLING_WINDOW, min_periods=1).mean()
    plt.figure(figsize=(14, 5))
    plt.plot(df["start_time"], df["health_indicator"], alpha=0.3, label="Health Indicator")
    plt.plot(df["start_time"], df["health_indicator_rolling"], linewidth=2, label=f"HI Rolling Mean ({HI_ROLLING_WINDOW})")
    plt.title("베어링 Health Indicator 추세")
    plt.xlabel("Time")
    plt.ylabel("Health Indicator")
    plt.legend()
    finish_plot(hi_dir / "health_indicator_trend.png")
    df[["start_time", "health_indicator", "health_indicator_rolling"]].to_csv(hi_dir / "health_indicator.csv", index=False)
    print("\nHealth Indicator 저장 완료")

if __name__ == "__main__":
    run()
