# ============================================================
# 08_Final_Timeline_Comparison.py
#
# Moving Z-score 열화 시작
# 교체 권고
# Isolation Forest 이상 시작
#
# 세 시점을 한 번에 비교
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 한글 폰트
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 결과 파일 경로
# ============================================================

ZSCORE_PATH = "results/moving_zscore/" "moving_zscore_summary.csv"

REPLACEMENT_PATH = "results/replacement_criteria/" "replacement_summary.csv"

IF_PATH = (
    "results/isolation_forest_validation/" "isolation_forest_validation_summary.csv"
)


# ============================================================
# 3. 저장 폴더
# ============================================================

SAVE_DIR = "results/final_timeline_comparison"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 4. 결과 불러오기
# ============================================================

zscore_df = pd.read_csv(ZSCORE_PATH, parse_dates=["degradation_start"])


replacement_df = pd.read_csv(
    REPLACEMENT_PATH,
    parse_dates=[
        "degradation_start",
        "replacement_time",
        "dataset_end_time",
    ],
)


if_df = pd.read_csv(
    IF_PATH,
    parse_dates=[
        "zscore_start",
        "if_start",
    ],
)


# ============================================================
# 5. 필요한 컬럼만 선택
# ============================================================

zscore_df = zscore_df[
    [
        "dataset",
        "degradation_start",
        "degradation_position_pct",
    ]
].copy()


replacement_df = replacement_df[
    [
        "dataset",
        "replacement_time",
        "replacement_position_pct",
        "dataset_end_time",
    ]
].copy()


if_df = if_df[
    [
        "dataset",
        "if_start",
        "if_position_pct",
    ]
].copy()


# ============================================================
# 6. 하나로 합치기
# ============================================================

final_df = zscore_df.merge(replacement_df, on="dataset", how="left").merge(
    if_df, on="dataset", how="left"
)


# ============================================================
# 7. 시간 차이 계산
# ============================================================

final_df["degradation_to_replacement"] = (
    final_df["replacement_time"] - final_df["degradation_start"]
)


final_df["replacement_to_if"] = final_df["if_start"] - final_df["replacement_time"]


final_df["degradation_to_if"] = final_df["if_start"] - final_df["degradation_start"]


final_df["degradation_to_end"] = (
    final_df["dataset_end_time"] - final_df["degradation_start"]
)


final_df["replacement_to_end"] = (
    final_df["dataset_end_time"] - final_df["replacement_time"]
)


final_df["if_to_end"] = final_df["dataset_end_time"] - final_df["if_start"]


# ============================================================
# 8. 수명 위치 차이
# ============================================================

final_df["deg_to_rep_pct"] = (
    final_df["replacement_position_pct"] - final_df["degradation_position_pct"]
)


final_df["rep_to_if_pct"] = (
    final_df["if_position_pct"] - final_df["replacement_position_pct"]
)


final_df["deg_to_if_pct"] = (
    final_df["if_position_pct"] - final_df["degradation_position_pct"]
)


# ============================================================
# 9. 출력
# ============================================================

print("\n" + "=" * 90)
print("최종 상태 변화 시점 비교")
print("=" * 90)


display_cols = [
    "dataset",
    "degradation_start",
    "degradation_position_pct",
    "replacement_time",
    "replacement_position_pct",
    "if_start",
    "if_position_pct",
    "dataset_end_time",
]


print(final_df[display_cols])


print("\n" + "=" * 90)
print("단계별 시간 차이")
print("=" * 90)


print(
    final_df[
        [
            "dataset",
            "degradation_to_replacement",
            "replacement_to_if",
            "degradation_to_if",
            "degradation_to_end",
            "replacement_to_end",
            "if_to_end",
        ]
    ]
)


# ============================================================
# 10. 저장
# ============================================================

final_df.to_csv(f"{SAVE_DIR}/" "final_timeline_comparison.csv", index=False)


# ============================================================
# 11. 수명 위치 비교 그래프
#
# dataset별:
# 열화 시작
# 교체 권고
# IF 이상 시작
# ============================================================

position_df = final_df[
    [
        "dataset",
        "degradation_position_pct",
        "replacement_position_pct",
        "if_position_pct",
    ]
].copy()


position_df = position_df.set_index("dataset")


position_df.columns = [
    "Moving Z-score 열화 시작",
    "교체 권고",
    "Isolation Forest 이상 시작",
]


# ============================================================
# 12. 데이터셋별 개별 그래프
# ============================================================

for dataset_name in position_df.index:

    row = position_df.loc[dataset_name]

    plt.figure(figsize=(10, 4))

    x = [
        "열화 시작",
        "교체 권고",
        "IF 이상 시작",
    ]

    y = [
        row["Moving Z-score 열화 시작"],
        row["교체 권고"],
        row["Isolation Forest 이상 시작"],
    ]

    plt.plot(x, y, marker="o")

    # --------------------------------------------------------
    # 값 표시
    # --------------------------------------------------------

    for label, value in zip(x, y):

        if pd.notna(value):

            plt.text(label, value + 1, f"{value:.2f}%", ha="center")

    plt.ylim(0, 105)

    plt.ylabel("전체 수명 위치 (%)")

    plt.title(f"{dataset_name} 상태 변화 시점 비교")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/" f"{dataset_name}_timeline_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ============================================================
# 13. 세 데이터셋 전체 비교 그래프
# ============================================================

plt.figure(figsize=(12, 6))


for dataset_name in position_df.index:

    row = position_df.loc[dataset_name]

    plt.plot(
        [
            "Moving Z-score",
            "교체 권고",
            "Isolation Forest",
        ],
        [
            row["Moving Z-score 열화 시작"],
            row["교체 권고"],
            row["Isolation Forest 이상 시작"],
        ],
        marker="o",
        label=dataset_name,
    )


plt.ylabel("전체 수명 위치 (%)")


plt.title("Set1 / Set2 / Set3 상태 변화 단계 비교")


plt.ylim(0, 105)


plt.grid(axis="y", alpha=0.3)


plt.legend()


plt.tight_layout()


plt.savefig(
    f"{SAVE_DIR}/" "all_dataset_timeline_comparison.png", dpi=300, bbox_inches="tight"
)


plt.show()
plt.close()


# ============================================================
# 14. 단계 순서 확인
# ============================================================

print("\n" + "=" * 90)
print("단계 순서 확인")
print("=" * 90)


for _, row in final_df.iterrows():

    dataset_name = row["dataset"]

    deg = row["degradation_position_pct"]

    rep = row["replacement_position_pct"]

    if_pos = row["if_position_pct"]

    print(f"\n{dataset_name}")

    print(f"열화 시작: " f"{deg:.2f}%")

    print(f"교체 권고: " f"{rep:.2f}%")

    print(f"IF 이상 시작: " f"{if_pos:.2f}%")

    # --------------------------------------------------------
    # 순서 해석
    # --------------------------------------------------------

    if deg < rep < if_pos:

        print("순서: 초기 열화" " → 교체 권고" " → 다변량 강한 이상")

    elif deg < if_pos < rep:

        print("순서: 초기 열화" " → IF 이상" " → 교체 권고")

    else:

        print("순서가 데이터셋별로 다름")


# ============================================================
# 15. 완료
# ============================================================

print("\n" + "=" * 90)
print("08 최종 비교 분석 완료")
print("=" * 90)

# Robust Moving Z-score는 조기 열화 감지용, 교체 기준은 실제 유지보수 의사결정용,
# Isolation Forest는 진행된 다변량 이상 확인용으로 역할이 분리되었다.
