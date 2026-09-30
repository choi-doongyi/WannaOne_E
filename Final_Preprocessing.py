# ============================================================
# 04_Final_Preprocessing.py
# 베어링 진동 특징 데이터 - 최종 전처리
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# 1. 데이터 경로
# ============================================================

DATASETS = {
    "set1": "data/set1_processed.csv",
    "set2": "data/set2_processed.csv",
    "set3": "data/set3_processed.csv",
}


# ============================================================
# 2. 저장 폴더
# ============================================================

SAVE_DIR = "data/model_ready"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 3. 세트별 최종 전처리
# ============================================================

for dataset_name, file_path in DATASETS.items():

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    # ========================================================
    # 데이터 불러오기
    # ========================================================

    df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])

    # ========================================================
    # 시간순 정렬
    # ========================================================

    df = df.sort_values("start_time").reset_index(drop=True)

    print("\n원본 Shape")
    print(df.shape)

    # ========================================================
    # 4. 중복 행 제거
    # ========================================================

    duplicate_count = df.duplicated().sum()

    print("\n중복 행")
    print(duplicate_count)

    if duplicate_count > 0:

        df = df.drop_duplicates().reset_index(drop=True)

    # ========================================================
    # 5. 대표 Segment 길이 확인
    #
    # 대부분의 측정 segment가 가지는 n_samples를
    # 정상적인 측정 길이로 사용
    # ========================================================

    print("\nn_samples 분포")

    print(df["n_samples"].value_counts().sort_values(ascending=False).head(10))

    main_sample_size = df["n_samples"].mode().iloc[0]

    print("\n대표 Segment 길이:", main_sample_size)

    # ========================================================
    # 6. 대표 길이가 아닌 Segment 확인
    # ========================================================

    abnormal_segment_count = (df["n_samples"] != main_sample_size).sum()

    print("비정상 길이 Segment:", abnormal_segment_count)

    # ========================================================
    # 7. 대표 Segment 길이만 유지
    #
    # FFT / RMS 등을 동일한 조건에서 비교하기 위해
    # 같은 측정 길이의 segment만 모델링에 사용
    # ========================================================

    df = df[df["n_samples"] == main_sample_size].copy().reset_index(drop=True)

    print("Segment 길이 정리 후 Shape:", df.shape)

    # ========================================================
    # 8. 모델에 사용하지 않을 컬럼
    # ========================================================

    metadata_cols = [
        "dataset",
        "segment",
        "start_time",
        "end_time",
        "n_samples",
        "stage",
        "health_indicator",
        "health_indicator_rolling",
    ]

    # 실제 존재하는 컬럼만
    metadata_cols = [col for col in metadata_cols if col in df.columns]

    # ========================================================
    # 9. Feature 컬럼 선택
    #
    # 숫자형이며 metadata가 아닌 컬럼
    # ========================================================

    feature_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in metadata_cols
    ]

    print("\n초기 Feature 개수:", len(feature_cols))

    # ========================================================
    # 10. inf / -inf → NaN
    # ========================================================

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # ========================================================
    # 11. 전체가 NaN인 Feature 제거
    # ========================================================

    all_nan_cols = [col for col in feature_cols if df[col].isna().all()]

    print("\n전체 NaN 컬럼:", all_nan_cols)

    if all_nan_cols:

        df = df.drop(columns=all_nan_cols)

        feature_cols = [col for col in feature_cols if col not in all_nan_cols]

    # ========================================================
    # 12. 결측치 확인
    # ========================================================

    missing = df[feature_cols].isna().sum().sort_values(ascending=False)

    print("\n결측치가 있는 Feature")

    print(missing[missing > 0])

    # ========================================================
    # 13. 남은 결측치는 중앙값으로 처리
    #
    # 극단값이 많은 베어링 특징 데이터이므로
    # 평균보다 Median 사용
    # ========================================================

    for col in feature_cols:

        if df[col].isna().any():

            df[col] = df[col].fillna(df[col].median())

    # ========================================================
    # 14. 상수 Feature 제거
    #
    # 모든 행에서 같은 값이면 모델에 정보가 없음
    # ========================================================

    constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]

    print("\n상수 Feature:", constant_cols)

    if constant_cols:

        df = df.drop(columns=constant_cols)

        feature_cols = [col for col in feature_cols if col not in constant_cols]

    # ========================================================
    # 15. 최종 NaN / inf 확인
    # ========================================================

    final_nan = df[feature_cols].isna().sum().sum()

    final_inf = np.isinf(df[feature_cols]).sum().sum()

    print("\n최종 확인")

    print("NaN:", final_nan)

    print("inf:", final_inf)

    # ========================================================
    # 16. 모델용 Feature DataFrame
    # ========================================================

    X = df[feature_cols].copy()

    print("\n최종 Feature 개수:", len(feature_cols))

    print("최종 X Shape:", X.shape)

    # ========================================================
    # 17. 최종 전체 데이터 저장
    #
    # 시간 정보도 보존
    # 나중에 예측 결과와 연결하기 위해 필요
    # ========================================================

    final_path = f"{SAVE_DIR}/" f"{dataset_name}_model_ready.csv"

    df.to_csv(final_path, index=False)

    # ========================================================
    # 18. Feature 목록 저장
    # ========================================================

    feature_path = f"{SAVE_DIR}/" f"{dataset_name}_feature_list.csv"

    pd.DataFrame({"feature": feature_cols}).to_csv(feature_path, index=False)

    print("\n저장 완료")

    print("Model Ready:", final_path)

    print("Feature List:", feature_path)


print("\n" + "=" * 60)
print("최종 전처리 완료")
print("=" * 60)
