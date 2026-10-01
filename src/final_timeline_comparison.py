import pandas as pd
import matplotlib.pyplot as plt
from src.common import ensure_dir, finish_plot, setup_plot_font
from src.config import FINAL_TIMELINE_DIR, ISOLATION_FOREST_SUMMARY, MOVING_ZSCORE_SUMMARY, REPLACEMENT_SUMMARY

def run():
    setup_plot_font()
    ensure_dir(FINAL_TIMELINE_DIR)
    zscore_df = pd.read_csv(MOVING_ZSCORE_SUMMARY, parse_dates=["degradation_start"])
    replacement_df = pd.read_csv(REPLACEMENT_SUMMARY, parse_dates=["degradation_start", "replacement_time", "dataset_end_time"])
    if_df = pd.read_csv(ISOLATION_FOREST_SUMMARY, parse_dates=["zscore_start", "if_start"])
    zscore_df = zscore_df[["dataset", "degradation_start", "degradation_position_pct"]].copy()
    replacement_df = replacement_df[["dataset", "replacement_time", "replacement_position_pct", "dataset_end_time"]].copy()
    if_df = if_df[["dataset", "if_start", "if_position_pct"]].copy()
    final_df = zscore_df.merge(replacement_df, on="dataset", how="left").merge(if_df, on="dataset", how="left")
    final_df["degradation_to_replacement"] = final_df["replacement_time"] - final_df["degradation_start"]
    final_df["replacement_to_if"] = final_df["if_start"] - final_df["replacement_time"]
    final_df["degradation_to_if"] = final_df["if_start"] - final_df["degradation_start"]
    final_df["degradation_to_end"] = final_df["dataset_end_time"] - final_df["degradation_start"]
    final_df["replacement_to_end"] = final_df["dataset_end_time"] - final_df["replacement_time"]
    final_df["if_to_end"] = final_df["dataset_end_time"] - final_df["if_start"]
    final_df["deg_to_rep_pct"] = final_df["replacement_position_pct"] - final_df["degradation_position_pct"]
    final_df["rep_to_if_pct"] = final_df["if_position_pct"] - final_df["replacement_position_pct"]
    final_df["deg_to_if_pct"] = final_df["if_position_pct"] - final_df["degradation_position_pct"]
    print("\n" + "=" * 90)
    print("최종 상태 변화 시점 비교")
    print("=" * 90)
    print(final_df[["dataset", "degradation_start", "degradation_position_pct", "replacement_time", "replacement_position_pct", "if_start", "if_position_pct", "dataset_end_time"]])
    print("\n" + "=" * 90)
    print("단계별 시간 차이")
    print("=" * 90)
    print(final_df[["dataset", "degradation_to_replacement", "replacement_to_if", "degradation_to_if", "degradation_to_end", "replacement_to_end", "if_to_end"]])
    final_df.to_csv(FINAL_TIMELINE_DIR / "final_timeline_comparison.csv", index=False)
    position_df = final_df[["dataset", "degradation_position_pct", "replacement_position_pct", "if_position_pct"]].copy().set_index("dataset")
    position_df.columns = ["Moving Z-score 열화 시작", "교체 권고", "Isolation Forest 이상 시작"]
    for dataset_name in position_df.index:
        row = position_df.loc[dataset_name]
        plt.figure(figsize=(10, 4))
        x = ["열화 시작", "교체 권고", "IF 이상 시작"]
        y = [row["Moving Z-score 열화 시작"], row["교체 권고"], row["Isolation Forest 이상 시작"]]
        plt.plot(x, y, marker="o")
        for label, value in zip(x, y):
            if pd.notna(value):
                plt.text(label, value + 1, f"{value:.2f}%", ha="center")
        plt.ylim(0, 105)
        plt.ylabel("전체 수명 위치 (%)")
        plt.title(f"{dataset_name} 상태 변화 시점 비교")
        plt.grid(axis="y", alpha=0.3)
        finish_plot(FINAL_TIMELINE_DIR / f"{dataset_name}_timeline_comparison.png")
    plt.figure(figsize=(12, 6))
    for dataset_name in position_df.index:
        row = position_df.loc[dataset_name]
        plt.plot(["Moving Z-score", "교체 권고", "Isolation Forest"], [row["Moving Z-score 열화 시작"], row["교체 권고"], row["Isolation Forest 이상 시작"]], marker="o", label=dataset_name)
    plt.ylabel("전체 수명 위치 (%)")
    plt.title("Set1 / Set2 / Set3 상태 변화 단계 비교")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    finish_plot(FINAL_TIMELINE_DIR / "all_dataset_timeline_comparison.png")
    print("\n" + "=" * 90)
    print("단계 순서 확인")
    print("=" * 90)
    for _, row in final_df.iterrows():
        deg, rep, if_pos = row["degradation_position_pct"], row["replacement_position_pct"], row["if_position_pct"]
        print(f"\n{row['dataset']}")
        print(f"열화 시작: {deg:.2f}%")
        print(f"교체 권고: {rep:.2f}%")
        print(f"IF 이상 시작: {if_pos:.2f}%")
        if deg < rep < if_pos:
            print("순서: 초기 열화 → 교체 권고 → 다변량 강한 이상")
        elif deg < if_pos < rep:
            print("순서: 초기 열화 → IF 이상 → 교체 권고")
        else:
            print("순서가 데이터셋별로 다름")

if __name__ == "__main__":
    run()
