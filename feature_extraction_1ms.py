"""인접한 타임스탬프 간격으로 IMS 측정 회차를 나누고 특징량을 계산합니다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


CHANNEL_PREFIX = "channel_"
SNAPSHOT_GAP = pd.Timedelta(seconds=1)


def calculate_features(
    signal: np.ndarray,
    channel_names: list[str],
    remove_dc: bool,
) -> dict[str, float]:
    """한 회차에서 채널별 RMS, 첨도, 최댓값, 파고율을 계산합니다."""
    # 옵션이 켜져 있으면 채널별 회차 평균을 빼서 기준선(DC 오프셋)을 제거합니다.
    # 기준선 주변의 진동 크기를 살펴보려는 경우에 유용합니다.
    if remove_dc:
        signal = signal - signal.mean(axis=0, keepdims=True)

    # RMS: 각 값을 제곱해 평균을 낸 뒤 제곱근을 계산합니다.
    # axis=0은 채널별로 따로 계산한다는 뜻입니다.
    rms = np.sqrt(np.mean(np.square(signal), axis=0))

    # 피어슨 첨도를 계산합니다. fisher=False이면 정규분포의 기준값은 3입니다.
    # axis=0으로 지정해 채널별로 독립적으로 계산합니다.
    kurtosis_values = kurtosis(
        signal,
        axis=0,
        fisher=False,
        bias=False,
    )

    # 최댓값(피크): 절댓값 기준으로 회차 안에서 가장 큰 진폭을 구합니다.
    peak = np.max(np.abs(signal), axis=0)

    # 파고율: 최댓값(피크)을 RMS로 나눕니다.
    # RMS가 0이면 나눗셈을 하지 않고 NaN을 반환합니다.
    crest_factor = np.divide(
        peak,
        rms,
        out=np.full_like(peak, np.nan, dtype=np.float64),
        where=rms != 0,
    )

    # 채널별 특징량을 각각 별도의 컬럼에 저장합니다.
    row: dict[str, float] = {}
    for index, channel in enumerate(channel_names):
        row[f"{channel}_rms"] = float(rms[index])
        row[f"{channel}_kurtosis"] = float(kurtosis_values[index])
        row[f"{channel}_peak"] = float(peak[index])
        row[f"{channel}_crest_factor"] = float(crest_factor[index])

    return row


def create_feature_dataset(
    input_path: Path,
    output_path: Path,
    remove_dc: bool,
) -> pd.DataFrame:
    """큰 CSV를 청크로 읽고 타임스탬프 간격에 따라 회차별 특징량을 만듭니다."""
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 CSV 파일을 찾을 수 없습니다: {input_path}")

    # 헤더만 읽어 진동 채널 컬럼명을 확인합니다.
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    if "timestamp" not in columns:
        raise ValueError("입력 CSV에 'timestamp' 컬럼이 있어야 합니다.")

    channel_names = [
        name for name in columns if name.startswith(CHANNEL_PREFIX)
    ]
    if not channel_names:
        raise ValueError("channel_1과 같은 채널 컬럼을 찾을 수 없습니다.")

    # 청크 경계에서 한 회차가 나뉠 수도 있으므로, 아래에서 이전 청크의
    # 마지막 미완성 회차를 보관했다가 다음 청크의 데이터와 이어 붙입니다.
    chunk_size = 100_000
    output_rows: list[dict[str, object]] = []
    next_snapshot_number = 1
    pending_values: np.ndarray | None = None
    pending_start_time: pd.Timestamp | None = None
    previous_timestamp: np.datetime64 | None = None

    def add_to_pending(segment: np.ndarray, start_time: np.datetime64) -> None:
        """현재 회차의 신호 조각을 저장하거나 기존 조각에 이어 붙입니다."""
        nonlocal pending_values, pending_start_time
        if pending_values is None:
            pending_values = segment.copy()
            pending_start_time = pd.Timestamp(start_time)
        else:
            pending_values = np.concatenate((pending_values, segment), axis=0)

    def finish_pending_snapshot() -> None:
        """현재까지 모은 회차의 특징량을 계산해 결과 목록에 추가합니다."""
        nonlocal pending_values, pending_start_time, next_snapshot_number
        if pending_values is None or pending_start_time is None:
            return

        row: dict[str, object] = {
            "snapshot": next_snapshot_number,
            "timestamp": pending_start_time,
            "sample_count": len(pending_values),
        }
        row.update(
            calculate_features(pending_values, channel_names, remove_dc)
        )
        output_rows.append(row)
        next_snapshot_number += 1
        pending_values = None
        pending_start_time = None

    for chunk in pd.read_csv(
        input_path,
        usecols=["timestamp", *channel_names],
        chunksize=chunk_size,
    ):
        # timestamp를 날짜·시간 형식으로 변환하고 채널 값은 숫자 배열로 가져옵니다.
        timestamps = pd.to_datetime(chunk["timestamp"], errors="raise").to_numpy()
        values = chunk[channel_names].to_numpy(dtype=np.float64)

        # 청크 첫 행과 이전 청크 마지막 행 사이가 1초보다 크면 회차가 바뀐 것입니다.
        if (
            pending_values is not None
            and previous_timestamp is not None
            and abs(timestamps[0] - previous_timestamp) > SNAPSHOT_GAP
        ):
            finish_pending_snapshot()

        # 청크 내부에서 인접 타임스탬프 간격이 1초를 초과하는 위치를 찾습니다.
        gap_positions = (
            np.flatnonzero(np.abs(np.diff(timestamps)) > SNAPSHOT_GAP) + 1
        )

        # 발견한 각 경계 전까지를 한 회차에 넣고, 경계에서 현재 회차를 마칩니다.
        segment_start = 0
        for boundary in gap_positions:
            add_to_pending(
                values[segment_start:boundary],
                timestamps[segment_start],
            )
            finish_pending_snapshot()
            segment_start = int(boundary)

        # 청크 마지막 조각은 다음 청크로 이어질 수 있으므로 우선 보관합니다.
        add_to_pending(values[segment_start:], timestamps[segment_start])
        previous_timestamp = timestamps[-1]

        print(f"{next_snapshot_number - 1}개 회차 처리 완료")

    # 파일 끝에 남아 있는 마지막 회차도 결과에 추가합니다.
    finish_pending_snapshot()

    if not output_rows:
        raise ValueError("입력 CSV에서 회차를 찾지 못했습니다.")

    # 회차별 결과를 데이터프레임으로 만들고 CSV 파일로 저장합니다.
    result = pd.DataFrame(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"저장 경로: {output_path}")
    print(f"회차 수: {len(result):,}; 컬럼 수: {len(result.columns)}")
    print(f"회차별 평균값(DC 오프셋) 제거 여부: {remove_dc}")
    return result


def create_all_feature_datasets(
    data_dir: Path,
    remove_dc: bool,
) -> dict[int, pd.DataFrame]:
    """1·2·3차 실험 CSV를 차례로 처리해 각각 특징량 CSV로 저장합니다."""
    results: dict[int, pd.DataFrame] = {}

    for experiment_number in (1, 2, 3):
        input_path = data_dir / f"test_set_{experiment_number}_1ms.csv"
        output_path = data_dir / f"test_set_{experiment_number}_snapshot_features.csv"

        print(f"\n{experiment_number}차 실험 처리 시작")
        results[experiment_number] = create_feature_dataset(
            input_path=input_path,
            output_path=output_path,
            remove_dc=remove_dc,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "인접 타임스탬프 간격이 1초를 초과하면 새 회차로 나눠 "
            "채널별 특징량을 계산합니다."
        )
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="data 폴더의 1·2·3차 실험 CSV를 모두 처리합니다.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/test_set_1_1ms.csv"),
        help="입력 밀리초 CSV 경로 (기본값: data/test_set_1_1ms.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test_set_1_snapshot_features.csv"),
        help="특징량 결과 CSV 경로",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="특징량 계산 전에 회차별 채널 평균을 빼지 않습니다.",
    )
    args = parser.parse_args()

    if args.all:
        # 실험별 입력 파일과 출력 파일 경로는 data 폴더에서 자동으로 구성합니다.
        create_all_feature_datasets(
            data_dir=Path("data"),
            remove_dc=not args.keep_dc,
        )
    else:
        # --all을 지정하지 않으면 --input과 --output으로 받은 파일 하나만 처리합니다.
        create_feature_dataset(
            input_path=args.input,
            output_path=args.output,
            remove_dc=not args.keep_dc,
        )


if __name__ == "__main__":
    main()
