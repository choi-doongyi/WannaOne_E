import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(
    "data/set1_processed.csv", parse_dates=["start_time", "end_time"]
)  # 두 컬럼을 날짜/시간 자료형으로 바로 읽어라

df = df.sort_values("start_time").reset_index(drop=True)


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
# ============================================================
# 2. 이미지 저장 폴더 생성
# ============================================================

SAVE_DIR = "results/eda/set1/time_trend"  # 저장할 폴더 위치

os.makedirs(SAVE_DIR, exist_ok=True)  # 만약 폴더가 있더라도 오류 X


# ============================================================
# 3. 확인할 특징
# ============================================================

feature_keywords = [
    "rms",  # 제곱평균제곱근 신호의 전체적인 크기/에너지 수준진동이 전반적으로 얼마나 강한가
    "kurtosis",  # 첨도 순간적으로 튀는 값이 얼마나 많은지충격성 이상이 있는가
    "crest_factor",  # 최대 피크가 RMS보다 얼마나 큰지순간 충격이 얼마나 강한가
    "peak_to_peak",  # 최대값 - 최소값전체 진폭 범위가 얼마나 큰가
    "spectral_energy",  # 이 신호가 주파수 영역에서 전체적으로 얼마나 강한 에너지를 가지고 있는가, 주파수 성분 전체의 에너지주파수 영역에서 진동 에너지가 얼마나 큰가
    "dominant_freq",  # 가장 강한 주파수 어떤 주파수에서 가장 크게 진동하는가
    "spectral_centroid",  # 주파수의 무게중심, 주파수의 무게중심에너지가 낮은/높은 주파수 중 어디에 몰렸는가
    "spectral_entropy",  # 주파수 에너지가 얼마나 퍼져 있는지특정 주파수에 집중됐나, 여러 주파수에 분산됐나
]


# ============================================================
# 4. 특징별 시간 추세
# ============================================================

for keyword in feature_keywords:

    cols = [col for col in df.columns if keyword in col.lower()]

    if len(cols) == 0:
        continue

    plt.figure(figsize=(14, 5))

    for col in cols:

        plt.plot(df["start_time"], df[col], label=col, alpha=0.8)

    plt.title(f"{keyword} 시간에 따른 변화")

    plt.xlabel("Time")
    plt.ylabel(keyword)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    # ========================================================
    # 이미지 저장
    # ========================================================

    save_path = f"{SAVE_DIR}/" f"{keyword}_time_trend.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print("저장:", save_path)

    # # 화면에도 출력
    # plt.show()

    # 메모리 정리
    plt.close()

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. 한글 폰트 설정
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/set1_processed.csv", parse_dates=["start_time", "end_time"])

df = df.sort_values("start_time").reset_index(drop=True)


# ============================================================
# 3. 저장 폴더 생성
# ============================================================

HIST_DIR = "results/eda/set1/distribution/histogram"
BOX_DIR = "results/eda/set1/distribution/boxplot"

os.makedirs(HIST_DIR, exist_ok=True)
os.makedirs(BOX_DIR, exist_ok=True)


# ============================================================
# 4. 분석할 특징 키워드
# ============================================================

feature_keywords = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_to_peak",
    "spectral_energy",
    "dominant_freq",
    "spectral_centroid",
    "spectral_entropy",
]


# ============================================================
# 5. 히스토그램 + KDE
# ============================================================

# for keyword in feature_keywords:

#     cols = [col for col in df.columns if keyword in col.lower()]

#     for col in cols:

#         plt.figure(figsize=(8, 4))

#         sns.histplot(data=df, x=col, bins=30, kde=True)

#         plt.title(f"{col} 분포")
#         plt.xlabel(col)
#         plt.ylabel("빈도")

#         plt.tight_layout()

#         save_path = f"{HIST_DIR}/{col}_hist.png"

#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#         print("저장:", save_path)

#         plt.show()
#         plt.close()


# ============================================================
# 6. 박스플롯
# ============================================================

# for keyword in feature_keywords:

#     cols = [col for col in df.columns if keyword in col.lower()]

#     for col in cols:

#         plt.figure(figsize=(6, 4))

#         sns.boxplot(y=df[col])

#         plt.title(f"{col} 박스플롯")
#         plt.ylabel(col)

#         plt.tight_layout()

#         save_path = f"{BOX_DIR}/{col}_boxplot.png"

#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#         print("저장:", save_path)

#         plt.show()
#         plt.close()


# ============================================================
# 7. 왜도 확인
# ============================================================

feature_cols = [
    col
    for col in df.select_dtypes(include="number").columns
    if col not in ["segment", "n_samples"]
]

skewness = df[feature_cols].skew().sort_values(ascending=False)

print("\n==============================")
print("왜도 확인")
print("==============================")

print(skewness)

# 왜도가 매우 큰 특징들이 다수 존재하며, 이는 일부 후반 구간에서 값이 급격히 증가하거나 감소하는 비대칭 분포를 의미한다.
# 베어링 열화 데이터 특성상 이러한 극단값은 제거 대상이 아니라 열화 신호 후보로 우선 확인한다.


top_skew_cols = skewness.abs().sort_values(ascending=False).head(10).index

# for col in top_skew_cols:
#     plt.figure(figsize=(14, 4))

#     plt.plot(df["start_time"], df[col])

#     plt.title(f"{col} 시간 추세")
#     plt.xlabel("Time")
#     plt.ylabel(col)

#     plt.tight_layout()
#     plt.show()


# ============================================================
# 1. 한글 폰트
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/set1_processed.csv", parse_dates=["start_time", "end_time"])


# ============================================================
# 3. 저장 폴더
# ============================================================

SAVE_DIR = "results/eda/set1/correlation"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 4. 수치형 특징 컬럼만 선택
#
# segment / n_samples는 제외
# ============================================================

feature_cols = [
    col
    for col in df.select_dtypes(include="number").columns
    if col not in ["segment", "n_samples"]
]


# ============================================================
# 5. 상관계수 계산
# ============================================================

corr = df[feature_cols].corr()


# ============================================================
# 채널별 상관관계 Heatmap
# ============================================================

CHANNEL_DIR = "results/eda/set1/correlation/by_channel"

os.makedirs(CHANNEL_DIR, exist_ok=True)


# channel_1 ~ channel_8
# for channel_num in range(1, 9):

#     channel_name = f"channel_{channel_num}"

#     # 해당 채널의 특징만 가져오기
#     channel_cols = [col for col in feature_cols if col.startswith(channel_name + "_")]

#     if len(channel_cols) < 2:
#         continue

#     channel_corr = df[channel_cols].corr()

#     plt.figure(figsize=(10, 8))

#     sns.heatmap(
#         channel_corr,
#         annot=True,  # 숫자 표시
#         fmt=".2f",  # 소수점 둘째자리
#         cmap="coolwarm",
#         center=0,
#         vmin=-1,
#         vmax=1,
#         square=True,
#     )

#     plt.title(f"{channel_name} 특징 상관관계")

#     plt.tight_layout()

#     save_path = f"{CHANNEL_DIR}/" f"{channel_name}_correlation.png"

#     plt.savefig(save_path, dpi=300, bbox_inches="tight")

#     print("저장:", save_path)

#     # plt.show()
#     plt.close()


# ============================================================
# 7. 상관계수가 높은 특징쌍 찾기
# ============================================================

high_corr_pairs = []

for i in range(len(corr.columns)):

    for j in range(i + 1, len(corr.columns)):

        value = corr.iloc[i, j]

        if abs(value) >= 0.9:

            high_corr_pairs.append(
                {
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": value,
                }
            )


high_corr_df = pd.DataFrame(high_corr_pairs)

if len(high_corr_df) > 0:

    high_corr_df["abs_correlation"] = high_corr_df["correlation"].abs()

    high_corr_df = high_corr_df.sort_values(
        "abs_correlation", ascending=False
    ).reset_index(drop=True)


print("\n==============================")
print("|상관계수| >= 0.9 특징쌍")
print("==============================")

print(high_corr_df.head(30))


# ============================================================
# 8. 높은 상관관계 목록 저장
# ============================================================

high_corr_df.to_csv(f"{SAVE_DIR}/high_correlation_pairs.csv", index=False)

# ============================================================
# 초기 / 중기 / 후기 비교
# ============================================================

n = len(df)

early_end = int(n * 0.3)
late_start = int(n * 0.7)

df["stage"] = "중기"
df.loc[: early_end - 1, "stage"] = "초기"
df.loc[late_start:, "stage"] = "후기"

SAVE_DIR = "results/eda/set1/degradation_stage"

os.makedirs(SAVE_DIR, exist_ok=True)

feature_keywords = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_to_peak",
    "spectral_energy",
    "dominant_freq",
    "spectral_centroid",
    "spectral_entropy",
]

# for keyword in feature_keywords:

#     cols = [col for col in df.columns if keyword in col.lower()]

#     for col in cols:

#         plt.figure(figsize=(7, 5))

#         sns.boxplot(data=df, x="stage", y=col, order=["초기", "중기", "후기"])

#         plt.title(f"{col} 초기/중기/후기 비교")
#         plt.xlabel("열화 단계")
#         plt.ylabel(col)

#         plt.tight_layout()

#         save_path = f"{SAVE_DIR}/" f"{col}_stage_boxplot.png"

#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

#         plt.show()
#         plt.close()


feature_cols = [
    col
    for col in df.select_dtypes(include="number").columns
    if col not in ["segment", "n_samples"]
]

stage_mean = df.groupby("stage")[feature_cols].mean().reindex(["초기", "중기", "후기"])

change_df = pd.DataFrame(
    {
        "early_mean": stage_mean.loc["초기"],
        "middle_mean": stage_mean.loc["중기"],
        "late_mean": stage_mean.loc["후기"],
    }
)

change_df["late_vs_early_ratio"] = change_df["late_mean"] / change_df["early_mean"]

change_df["difference"] = change_df["late_mean"] - change_df["early_mean"]

change_df = change_df.sort_values("late_vs_early_ratio", ascending=False)

change_df.to_csv(f"{SAVE_DIR}/stage_feature_change.csv")

print(change_df.head(20))
