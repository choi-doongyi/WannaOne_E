import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/set1_processed.csv", parse_dates=["start_time", "end_time"])

df = df.sort_values("start_time").reset_index(drop=True)


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
# ============================================================
# 2. 이미지 저장 폴더 생성
# ============================================================

SAVE_DIR = "results/eda/set1/time_trend"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 3. 확인할 특징
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

    # 화면에도 출력
    plt.show()

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

for keyword in feature_keywords:

    cols = [col for col in df.columns if keyword in col.lower()]

    for col in cols:

        plt.figure(figsize=(8, 4))

        sns.histplot(data=df, x=col, bins=30, kde=True)

        plt.title(f"{col} 분포")
        plt.xlabel(col)
        plt.ylabel("빈도")

        plt.tight_layout()

        save_path = f"{HIST_DIR}/{col}_hist.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        print("저장:", save_path)

        plt.show()
        plt.close()


# ============================================================
# 6. 박스플롯
# ============================================================

for keyword in feature_keywords:

    cols = [col for col in df.columns if keyword in col.lower()]

    for col in cols:

        plt.figure(figsize=(6, 4))

        sns.boxplot(y=df[col])

        plt.title(f"{col} 박스플롯")
        plt.ylabel(col)

        plt.tight_layout()

        save_path = f"{BOX_DIR}/{col}_boxplot.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        print("저장:", save_path)

        plt.show()
        plt.close()


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
