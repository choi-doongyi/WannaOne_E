import numpy as np
import pandas as pd
from src.config import RAW_DATASETS

def analyze_dataset(dataset_name, file_path):
    print("\n" + "=" * 70)
    print(f"{dataset_name} - Raw Data DDA")
    print("=" * 70)
    df = pd.read_csv(file_path)
    print("\n컬럼")
    print(df.columns.tolist())
    print("\nShape")
    print(df.shape)
    print("\nInfo")
    df.info()
    print("\nDescribe")
    print(df.describe().T)
    print("\n중복 행")
    print(df.duplicated().sum())
    print("\n컬럼별 고유값 개수")
    print(df.nunique())
    print("\n결측치")
    missing = df.isna().sum()
    print(missing[missing > 0] if (missing > 0).any() else "없음")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("\n시간 범위")
    print("start:", df["timestamp"].min())
    print("end  :", df["timestamp"].max())
    diff = df["timestamp"].sort_values().diff()
    print("\n시간 간격 상위")
    print(diff.value_counts().head(10))
    numeric_cols = df.select_dtypes(include="number").columns
    print("\n무한대 값")
    print(df[numeric_cols].isin([np.inf, -np.inf]).sum())
    df = df.sort_values("timestamp").reset_index(drop=True)
    diff = df["timestamp"].diff()
    gaps = diff[diff > pd.Timedelta(milliseconds=1)]
    print("\n1ms 초과 Gap 개수:", len(gaps))
    print("\nGap 크기 상위")
    print(gaps.value_counts().head(10))
    print("\n측정 Segment 수:", len(gaps) + 1)

def run():
    for dataset_name, file_path in RAW_DATASETS.items():
        analyze_dataset(dataset_name, file_path)

if __name__ == "__main__":
    run()
