import pandas as pd
import numpy as np

DATASETS = {
    "set1": "data/set1_features.csv",
    "set2": "data/set2_features.csv",
    "set3": "data/set3_features.csv",
}


for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 50)
    print(dataset_name)
    print("=" * 50)

    df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])

    # ========================================================
    # 1. 기본 구조 확인
    # ========================================================

    print("\nShape")
    print(df.shape)

    print("\n컬럼")
    print(df.columns.tolist())

    print("\n데이터 타입")
    print(df.dtypes)

    # ========================================================
    # 2. 중복 확인
    # ========================================================

    duplicate_count = df.duplicated().sum()

    print("\n중복 행 개수")
    print(duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # ========================================================
    # 3. 결측치 확인
    # ========================================================

    missing_count = df.isna().sum()

    print("\n결측치가 있는 컬럼")
    print(missing_count[missing_count > 0].sort_values(ascending=False))

    # ========================================================
    # 4. inf / -inf 확인
    # ========================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    inf_count = np.isinf(df[numeric_cols]).sum()

    print("\n무한대 값이 있는 컬럼")
    print(inf_count[inf_count > 0])

    # inf를 NaN으로 변환
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # ========================================================
    # 5. segment sample 수 확인
    # ========================================================

    print("\nn_samples 분포")
    print(df["n_samples"].value_counts().sort_index())

    # ========================================================
    # 6. 너무 짧은 segment 확인
    #
    # 바로 삭제하지 않고 우선 확인만
    # ========================================================

    print("\nn_samples 요약")
    print(df["n_samples"].describe())

    # ========================================================
    # 7. 상수 컬럼 확인
    # ========================================================

    feature_cols = [col for col in numeric_cols if col not in ["segment", "n_samples"]]

    constant_cols = [col for col in feature_cols if df[col].nunique(dropna=True) <= 1]

    print("\n상수 컬럼")
    print(constant_cols)

    # 상수 컬럼 제거
    if constant_cols:
        df = df.drop(columns=constant_cols)

    # ========================================================
    # 8. 최종 결측치 재확인
    # ========================================================

    print("\n최종 결측치 개수")
    print(df.isna().sum().sum())

    # ========================================================
    # 9. 저장
    # ========================================================

    output_path = f"data/{dataset_name}_processed.csv"

    df.to_csv(output_path, index=False)

    print("\n저장 완료")
    print(output_path)


import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv("data/set1_processed.csv", parse_dates=["start_time", "end_time"])

df = df.sort_values("start_time").reset_index(drop=True)


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
