import numpy as np
import pandas as pd
from src.common import ensure_dir
from src.config import FEATURE_LIST_PATHS, MODEL_READY_DATASETS, MODEL_READY_DIR, PROCESSED_DATASETS

def run():
    ensure_dir(MODEL_READY_DIR)
    for dataset_name, file_path in PROCESSED_DATASETS.items():
        print("\n" + "=" * 60)
        print(dataset_name)
        print("=" * 60)
        df = pd.read_csv(file_path, parse_dates=["start_time", "end_time"])
        df = df.sort_values("start_time").reset_index(drop=True)
        print("\n원본 Shape")
        print(df.shape)
        duplicate_count = df.duplicated().sum()
        print("\n중복 행")
        print(duplicate_count)
        if duplicate_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        print("\nn_samples 분포")
        print(df["n_samples"].value_counts().sort_values(ascending=False).head(10))
        main_sample_size = df["n_samples"].mode().iloc[0]
        print("\n대표 Segment 길이:", main_sample_size)
        abnormal_segment_count = (df["n_samples"] != main_sample_size).sum()
        print("비정상 길이 Segment:", abnormal_segment_count)
        df = df[df["n_samples"] == main_sample_size].copy().reset_index(drop=True)
        print("Segment 길이 정리 후 Shape:", df.shape)
        metadata_cols = ["dataset", "segment", "start_time", "end_time", "n_samples", "stage", "health_indicator", "health_indicator_rolling"]
        metadata_cols = [col for col in metadata_cols if col in df.columns]
        feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in metadata_cols]
        print("\n초기 Feature 개수:", len(feature_cols))
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        all_nan_cols = [col for col in feature_cols if df[col].isna().all()]
        print("\n전체 NaN 컬럼:", all_nan_cols)
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            feature_cols = [col for col in feature_cols if col not in all_nan_cols]
        missing = df[feature_cols].isna().sum().sort_values(ascending=False)
        print("\n결측치가 있는 Feature")
        print(missing[missing > 0])
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        constant_cols = [col for col in feature_cols if df[col].nunique() <= 1]
        print("\n상수 Feature:", constant_cols)
        if constant_cols:
            df = df.drop(columns=constant_cols)
            feature_cols = [col for col in feature_cols if col not in constant_cols]
        final_nan = df[feature_cols].isna().sum().sum()
        final_inf = np.isinf(df[feature_cols]).sum().sum()
        print("\n최종 확인")
        print("NaN:", final_nan)
        print("inf:", final_inf)
        X = df[feature_cols].copy()
        print("\n최종 Feature 개수:", len(feature_cols))
        print("최종 X Shape:", X.shape)
        final_path = MODEL_READY_DATASETS[dataset_name]
        df.to_csv(final_path, index=False)
        pd.DataFrame({"feature": feature_cols}).to_csv(FEATURE_LIST_PATHS[dataset_name], index=False)
        print("\n저장 완료")
        print("Model Ready:", final_path)
        print("Feature List:", FEATURE_LIST_PATHS[dataset_name])
    print("\n" + "=" * 60)
    print("최종 전처리 완료")
    print("=" * 60)

if __name__ == "__main__":
    run()
