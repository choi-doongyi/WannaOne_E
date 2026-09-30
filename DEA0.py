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
