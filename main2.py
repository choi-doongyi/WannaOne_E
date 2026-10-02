import argparse
import time
from src.dda import run as run_dda
from src.feature_extraction import run as run_feature_extraction
from src.feature_dda import run as run_feature_dda
from src.eda import run as run_eda
from src.time_series_eda import run as run_time_series_eda
from src.final_preprocessing import run as run_final_preprocessing
from src.moving_zscore_detection import run as run_moving_zscore
from src.replacement_criteria import run as run_replacement
from src.isolation_forest_validation import run as run_isolation_forest
from src.final_timeline_comparison import run as run_final_timeline
from src.fft_analysis import run as run_fft

STEPS = [
    ("DDA", run_dda, False),
    ("Feature 추출", run_feature_extraction, False),
    ("Feature DDA", run_feature_dda, False),
    ("EDA", run_eda, False),
    ("Time-Series EDA", run_time_series_eda, False),
    ("최종 전처리", run_final_preprocessing, False),
    ("Moving Z-score 열화 탐지", run_moving_zscore, False),
    ("교체 권고 기준", run_replacement, False),
    ("Isolation Forest 검증", run_isolation_forest, False),
    ("최종 Timeline 비교", run_final_timeline, False),
    ("FFT 선택과제", run_fft, True),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="대형 회전기 베어링 열화 추적 전체 분석 파이프라인"
    )
    p.add_argument("--no-fft", action="store_true", help="선택과제 FFT 분석 제외")
    p.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=range(1, 12),
        metavar="N",
        help="N번째 단계부터 실행",
    )
    return p.parse_args()


def main():
    args = parse_args()
    selected = []
    for idx, (name, func, optional) in enumerate(STEPS, 1):
        if idx < args.from_step:
            continue
        if optional and args.no_fft:
            continue
        selected.append((idx, name, func))

    print("\n" + "=" * 70)
    print("대형 회전기 베어링 열화 추적 분석 시작")
    print("=" * 70)
    total_start = time.perf_counter()

    for idx, name, func in selected:
        print("\n" + "=" * 70)
        print(f"[{idx}/11] {name}")
        print("=" * 70)
        t0 = time.perf_counter()
        try:
            func()
        except FileNotFoundError as exc:
            print(f"\n[실행 중단] 필요한 파일을 찾지 못했습니다: {exc}")
            print("이전 단계가 완료됐는지, src/config.py 경로가 맞는지 확인하세요.")
            raise
        print(f"\n{name} 완료: {time.perf_counter()-t0:.1f}초")

    print("\n" + "=" * 70)
    print(f"전체 분석 완료 / 총 실행 시간: {time.perf_counter()-total_start:.1f}초")
    print("=" * 70)


if __name__ == "__main__":
    main()
