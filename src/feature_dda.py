import numpy as np
import pandas as pd
from src.config import FEATURE_DATASETS, PROCESSED_DATASETS

def run():
    for dataset_name, file_path in FEATURE_DATASETS.items():
        print("\n" + "=" * 60)
        print(dataset_name)
        print("=" * 60)
        df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
        print("\nShape")
        print(df.shape)
        print("\n컬럼")
        print(df.columns.tolist())
        print("\n데이터 타입")
        print(df.dtypes)
        duplicate_count = df.duplicated().sum()
        print("\n중복 행 개수")
        print(duplicate_count)
        if duplicate_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        missing_count = df.isna().sum()
        print("\n결측치가 있는 컬럼")
        print(missing_count[missing_count > 0].sort_values(ascending=False))
        numeric_cols = df.select_dtypes(include=np.number).columns
        inf_count = np.isinf(df[numeric_cols]).sum()
        print("\n무한대 값이 있는 컬럼")
        print(inf_count[inf_count > 0])
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        print("\nn_samples 분포")
        print(df["n_samples"].value_counts().sort_index())
        print("\nn_samples 요약")
        print(df["n_samples"].describe())
        feature_cols = [col for col in numeric_cols if col not in ["segment", "n_samples"]]
        constant_cols = [col for col in feature_cols if df[col].nunique(dropna=True) <= 1]
        print("\n상수 컬럼")
        print(constant_cols)
        if constant_cols:
            df = df.drop(columns=constant_cols)
        print("\n최종 결측치 개수")
        print(df.isna().sum().sum())
        output_path = PROCESSED_DATASETS[dataset_name]
        df.to_csv(output_path, index=False)
        print("\n저장 완료")
        print(output_path)

if __name__ == "__main__":
    run()
