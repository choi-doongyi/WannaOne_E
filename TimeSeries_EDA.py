# ============================================================
# 03_TimeSeries_EDA.py
# 베어링 진동 데이터 - 시계열 EDA
# ============================================================

import os

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 한글 폰트 설정
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글폰트 설정
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/set1_processed.csv", parse_dates=["start_time", "end_time"])


# ============================================================
# 3. 시간순 정렬
# ============================================================

df = df.sort_values("start_time").reset_index(drop=True)


print("데이터 크기:", df.shape)

print("\n시간 범위")
print(df["start_time"].min(), "~", df["start_time"].max())


# ============================================================
# 4. 결과 저장 폴더
# ============================================================

SAVE_DIR = "results/eda/set1/time_series"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 5. 분석할 주요 특징
# ============================================================

feature_keywords = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_to_peak",
    "spectral_energy",
]


# ============================================================
# 6. Rolling Mean 설정
#
# 최근 20개 측정 segment 평균
# ============================================================

ROLLING_WINDOW = 20


# ============================================================
# 7. 특징별 Rolling Mean + 변화점 탐색
# ============================================================

change_point_results = []


for keyword in feature_keywords:

    cols = [col for col in df.columns if keyword in col.lower()]

    for col in cols:

        # ----------------------------------------------------
        # Rolling Mean
        # ----------------------------------------------------

        rolling_mean = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()

        # ----------------------------------------------------
        # Rolling Mean의 변화량
        # ----------------------------------------------------

        rolling_diff = rolling_mean.diff().abs()

        # ----------------------------------------------------
        # 가장 크게 변한 지점
        # ----------------------------------------------------

        change_idx = rolling_diff.idxmax()

        change_time = df.loc[change_idx, "start_time"]

        change_value = rolling_diff.loc[change_idx]

        # ----------------------------------------------------
        # 결과 저장
        # ----------------------------------------------------

        change_point_results.append(
            {
                "feature": col,
                "change_index": change_idx,
                "change_time": change_time,
                "change_value": change_value,
            }
        )

        print("\n==============================")
        print(col)
        print("==============================")

        print("가장 큰 변화 시점:", change_time)

        print("변화량:", change_value)

        # ====================================================
        # 시각화
        # ====================================================

        plt.figure(figsize=(14, 5))

        # 원본 특징값
        plt.plot(df["start_time"], df[col], alpha=0.35, label="원본")

        # Rolling Mean
        plt.plot(
            df["start_time"],
            rolling_mean,
            linewidth=2,
            label=f"Rolling Mean ({ROLLING_WINDOW})",
        )

        # 변화점
        plt.axvline(x=change_time, linestyle="--", label="가장 큰 변화점")

        plt.title(f"{col} 시계열 변화점 탐색")

        plt.xlabel("Time")

        plt.ylabel(col)

        plt.legend()

        plt.tight_layout()

        # ====================================================
        # 이미지 저장
        # ====================================================

        save_path = f"{SAVE_DIR}/" f"{col}_change_point.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        print("이미지 저장:", save_path)

        plt.show()

        plt.close()


# ============================================================
# 8. 변화점 결과 DataFrame
# ============================================================

change_point_df = pd.DataFrame(change_point_results)


# ============================================================
# 9. 변화량 기준 정렬
# ============================================================

change_point_df = change_point_df.sort_values(
    "change_value", ascending=False
).reset_index(drop=True)


print("\n==============================")
print("변화점 탐색 결과")
print("==============================")

print(change_point_df.head(20))


# ============================================================
# 10. 변화점 결과 저장
# ============================================================

change_point_df.to_csv(f"{SAVE_DIR}/change_point_results.csv", index=False)


print("\n변화점 결과 저장 완료")


# ============================================================
# 11. 특징별 변화점 시점 비교
# ============================================================

CLUSTER_DIR = "results/eda/set1/change_point_cluster"

os.makedirs(CLUSTER_DIR, exist_ok=True)


# change_point_df는 위 코드에서 이미 만들어짐
# 혹시 새로 실행한다면 아래처럼 불러오면 됨
#
# change_point_df = pd.read_csv(
#     "results/eda/set1/time_series/change_point_results.csv",
#     parse_dates=["change_time"]
# )


# ============================================================
# 12. 변화시점 정렬
# ============================================================

change_point_df["change_time"] = pd.to_datetime(change_point_df["change_time"])

change_point_sorted = change_point_df.sort_values("change_time").reset_index(drop=True)


print("\n==============================")
print("특징별 변화시점 정렬")
print("==============================")

print(change_point_sorted[["feature", "change_time", "change_value"]])


# ============================================================
# 13. 변화점 산점도
#
# x축 = 시간
# y축 = 특징
# ============================================================

plt.figure(figsize=(14, 10))

plt.scatter(change_point_sorted["change_time"], change_point_sorted["feature"])

plt.title("특징별 변화점 발생 시점")

plt.xlabel("Change Time")

plt.ylabel("Feature")

plt.xticks(rotation=45)

plt.tight_layout()


save_path = f"{CLUSTER_DIR}/" "change_point_timeline.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print("저장:", save_path)

plt.show()
plt.close()


# ============================================================
# 14. 변화시점 시간 단위 집계
#
# 여기서는 '일' 단위로 묶어서
# 같은 날짜에 몇 개 특징이 변했는지 확인
# ============================================================

change_point_sorted["change_date"] = change_point_sorted["change_time"].dt.floor("D")

change_count = (
    change_point_sorted.groupby("change_date").size().sort_values(ascending=False)
)


print("\n==============================")
print("날짜별 변화점 개수")
print("==============================")

print(change_count.head(20))


# ============================================================
# 15. 변화점 집중 시점 막대그래프
# ============================================================

change_count_plot = change_count.sort_index()

plt.figure(figsize=(14, 5))

plt.bar(change_count_plot.index, change_count_plot.values)

plt.title("날짜별 변화점 발생 개수")

plt.xlabel("Date")

plt.ylabel("변화 특징 개수")

plt.xticks(rotation=45)

plt.tight_layout()


save_path = f"{CLUSTER_DIR}/" "change_point_count_by_date.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print("저장:", save_path)

plt.show()
plt.close()


# ============================================================
# 16. 변화점이 가장 많이 몰린 날짜
# ============================================================

if len(change_count) > 0:

    peak_change_date = change_count.idxmax()

    peak_change_count = change_count.max()

    print("\n==============================")
    print("변화점 집중 시점")
    print("==============================")

    print("가장 많은 변화점이 발생한 날짜:", peak_change_date)

    print("해당 날짜 변화 특징 개수:", peak_change_count)

    # 해당 날짜에 변화한 특징 확인
    peak_features = change_point_sorted[
        change_point_sorted["change_date"] == peak_change_date
    ]

    print("\n해당 시점 변화 특징")

    print(peak_features[["feature", "change_time", "change_value"]])

    # CSV 저장
    peak_features.to_csv(f"{CLUSTER_DIR}/" "peak_change_features.csv", index=False)

# ============================================================
# 17. Health Indicator 생성
# ============================================================

from sklearn.preprocessing import StandardScaler

HI_DIR = "results/eda/set1/health_indicator"

os.makedirs(HI_DIR, exist_ok=True)


# ============================================================
# 18. HI에 사용할 특징 선택
#
# 우선 베어링 열화에 대표적인 특징들 사용
# ============================================================

hi_keywords = ["rms", "kurtosis", "crest_factor", "peak_to_peak", "spectral_energy"]


hi_cols = [
    col for col in df.columns if any(keyword in col.lower() for keyword in hi_keywords)
]


print("\n==============================")
print("Health Indicator 사용 특징")
print("==============================")

print(hi_cols)


# ============================================================
# 19. 결측치 확인
# ============================================================

hi_data = df[hi_cols].copy()

print("\nHI 특징 결측치 개수")
print(hi_data.isna().sum().sum())


# 혹시 NaN이 있다면 중앙값으로 채움
hi_data = hi_data.fillna(hi_data.median())


# ============================================================
# 20. 특징 표준화
#
# 평균 = 0
# 표준편차 = 1
# ============================================================

hi_scaler = StandardScaler()

hi_scaled = hi_scaler.fit_transform(hi_data)


# ============================================================
# 21. Health Indicator 계산
#
# 모든 표준화 특징의 평균
# ============================================================

df["health_indicator"] = hi_scaled.mean(axis=1)


# ============================================================
# 22. HI Rolling Mean
# ============================================================

HI_ROLLING_WINDOW = 20

df["health_indicator_rolling"] = (
    df["health_indicator"].rolling(window=HI_ROLLING_WINDOW, min_periods=1).mean()
)


# ============================================================
# 23. Health Indicator 시각화
# ============================================================

plt.figure(figsize=(14, 5))


# 원본 HI
plt.plot(df["start_time"], df["health_indicator"], alpha=0.3, label="Health Indicator")


# Rolling HI
plt.plot(
    df["start_time"],
    df["health_indicator_rolling"],
    linewidth=2,
    label=f"HI Rolling Mean ({HI_ROLLING_WINDOW})",
)


plt.title("베어링 Health Indicator 추세")

plt.xlabel("Time")

plt.ylabel("Health Indicator")

plt.legend()

plt.tight_layout()


save_path = f"{HI_DIR}/" "health_indicator_trend.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")


print("저장:", save_path)

plt.show()

plt.close()


# ============================================================
# 24. 초기 / 중기 / 후기 HI 평균 비교
# ============================================================

if "stage" in df.columns:

    hi_stage_mean = (
        df.groupby("stage")["health_indicator"].mean().reindex(["초기", "중기", "후기"])
    )

    print("\n==============================")
    print("단계별 Health Indicator 평균")
    print("==============================")

    print(hi_stage_mean)


# ============================================================
# 25. HI 저장
# ============================================================

hi_output = df[["start_time", "health_indicator", "health_indicator_rolling"]].copy()


hi_output.to_csv(f"{HI_DIR}/" "health_indicator.csv", index=False)


print("\nHealth Indicator 저장 완료")
