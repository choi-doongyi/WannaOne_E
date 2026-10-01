from pathlib import Path
import pandas as pd

# 프로젝트 루트: project/src/config.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_READY_DIR = DATA_DIR / "model_ready"

# 전체 실행 시 그래프 창이 수십 개 뜨지 않도록 기본 False
SHOW_PLOTS = False
PLOT_FONT = "Malgun Gothic"

RAW_DATASETS = {
    "set1": DATA_DIR / "test_set_1_1ms.csv",
    "set2": DATA_DIR / "test_set_2_1ms.csv",
    "set3": DATA_DIR / "test_set_3_1ms.csv",
}
FEATURE_DATASETS = {k: DATA_DIR / f"{k}_features.csv" for k in RAW_DATASETS}
PROCESSED_DATASETS = {k: DATA_DIR / f"{k}_processed.csv" for k in RAW_DATASETS}
MODEL_READY_DATASETS = {
    k: MODEL_READY_DIR / f"{k}_model_ready.csv" for k in RAW_DATASETS
}
FEATURE_LIST_PATHS = {
    k: MODEL_READY_DIR / f"{k}_feature_list.csv" for k in RAW_DATASETS
}

FS = 1000
GAP_THRESHOLD = pd.Timedelta(seconds=1)

EDA_DATASET = "set1"
EDA_DIR = RESULTS_DIR / "eda" / EDA_DATASET

BASELINE_RATIO = 0.30
Z_THRESHOLD = 3.0
ZSCORE_ROLLING_WINDOW = 5
MIN_CHANNELS = 2
ZSCORE_CONSECUTIVE_COUNT = 3
ROBUST_SCALE = 0.6745
MOVING_ZSCORE_DIR = RESULTS_DIR / "moving_zscore"
MOVING_ZSCORE_SUMMARY = MOVING_ZSCORE_DIR / "moving_zscore_summary.csv"
MOVING_ZSCORE_DETAIL_PATHS = {
    k: MOVING_ZSCORE_DIR / k / f"{k}_moving_zscore_detail.csv" for k in RAW_DATASETS
}

SEVERE_Z_THRESHOLD = 5.0
MIN_SEVERE_CHANNELS = 2
MIN_FEATURE_TYPES = 2
REPLACEMENT_CONSECUTIVE_COUNT = 5
REPLACEMENT_FEATURE_TYPES = ["rms", "kurtosis", "max", "crest_factor"]
REPLACEMENT_DIR = RESULTS_DIR / "replacement_criteria"
REPLACEMENT_SUMMARY = REPLACEMENT_DIR / "replacement_summary.csv"

IF_N_ESTIMATORS = 300
IF_RANDOM_STATE = 42
IF_ANOMALY_PERCENTILE = 99
IF_ROLLING_WINDOW = 5
IF_ROLLING_ANOMALY_RATIO = 0.60
IF_CONSECUTIVE_COUNT = 3
ISOLATION_FOREST_DIR = RESULTS_DIR / "isolation_forest_validation"
ISOLATION_FOREST_SUMMARY = (
    ISOLATION_FOREST_DIR / "isolation_forest_validation_summary.csv"
)

FINAL_TIMELINE_DIR = RESULTS_DIR / "final_timeline_comparison"

FFT_COMPARE_SEGMENTS = 10
FFT_FREQ_BANDS = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
    (300, 350),
    (350, 400),
    (400, 450),
    (450, 500),
]
FFT_DIR = RESULTS_DIR / "fft_analysis"
