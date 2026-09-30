"""Extract time-domain features from NASA IMS raw snapshots.

Each extensionless source file is one acquisition. The 1st_test has 8 channels;
the 2nd_test and 3rd_test have 4 channels. Output has one row per snapshot.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis


TIMESTAMP_PATTERN = re.compile(r"\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}")
def feature_names(channel_number: int, channel_count: int) -> str:
    if channel_count == 8:
        bearing_number = (channel_number - 1) // 2 + 1
        within_bearing_channel = (channel_number - 1) % 2 + 1
    elif channel_count == 4:
        bearing_number = channel_number
        within_bearing_channel = 1
    else:
        raise ValueError(f"Unsupported IMS channel count: {channel_count}")
    return f"bearing_{bearing_number}_ch{within_bearing_channel}"


def extract_snapshot(path: Path) -> dict[str, float | pd.Timestamp | str]:
    timestamp = pd.to_datetime(
        path.name,
        format="%Y.%m.%d.%H.%M.%S",
        errors="raise",
    )
    with path.open("r", encoding="utf-8") as source:
        first_line = source.readline()
    channel_count = len(first_line.split())
    channels = [f"channel_{number}" for number in range(1, channel_count + 1)]
    if channel_count not in (4, 8):
        raise ValueError(f"Expected 4 or 8 channels in {path}, got {channel_count}")

    signal = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=channels,
        dtype=np.float64,
    )

    if signal.shape[1] != channel_count or signal.empty:
        raise ValueError(f"Unexpected channel layout in {path}")

    row: dict[str, float | pd.Timestamp | str] = {
        "timestamp": timestamp,
        "source_file": path.name,
    }

    for number, channel in enumerate(channels, start=1):
        values = signal[channel].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite samples found in {path}, {channel}")

        rms = float(np.sqrt(np.mean(np.square(values))))
        peak = float(np.max(np.abs(values)))
        prefix = feature_names(number, channel_count)

        row[f"{prefix}_rms"] = rms
        # Pearson kurtosis: a normal distribution has kurtosis 3.
        row[f"{prefix}_kurtosis"] = float(
            kurtosis(values, fisher=False, bias=False)
        )
        row[f"{prefix}_peak"] = peak
        row[f"{prefix}_crest_factor"] = peak / rms if rms else np.nan

    return row


def extract_dataset(input_dir: Path, output_path: Path) -> pd.DataFrame:
    files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and TIMESTAMP_PATTERN.fullmatch(path.name)
    )
    if not files:
        raise FileNotFoundError(
            f"No timestamp-named IMS snapshot files found under: {input_dir}"
        )

    rows = []
    for index, path in enumerate(files, start=1):
        rows.append(extract_snapshot(path))
        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} snapshots")

    result = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({result.shape[0]} rows, {result.shape[1]} columns)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/1st_test/1st_test"),
        help="Directory containing original extensionless IMS snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ims_test1_features.csv"),
        help="Path for the generated feature table (choose a separate file per test)",
    )
    args = parser.parse_args()
    extract_dataset(args.input_dir, args.output)


if __name__ == "__main__":
    main()
