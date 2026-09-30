# ============================================================
# 06_Replacement_Criteria.py
# Moving Z-score 기반 베어링 교체 권고 시점 탐지
#
# 교체 권고 기준:
# 1) 05번 열화 시작 이후만 분석
# 2) 서로 다른 채널 2개 이상에서 심각 이상
# 3) 서로 다른 특징 종류 2종 이상에서 심각 이상
# 4) 일정 segment 이상 연속 지속
# ============================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 한글 폰트
# ============================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 05번 결과 경로
# ============================================================

DETAIL_PATHS = {
    "set1": "results/moving_zscore/set1/set1_moving_zscore_detail.csv",
    "set2": "results/moving_zscore/set2/set2_moving_zscore_detail.csv",
    "set3": "results/moving_zscore/set3/set3_moving_zscore_detail.csv",
}


SUMMARY_PATH = "results/moving_zscore/" "moving_zscore_summary.csv"


# ============================================================
# 3. 저장 폴더
# ============================================================

SAVE_DIR = "results/replacement_criteria"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 4. 교체 권고 기준
# ============================================================

# 열화보다 더 강한 이상 기준
SEVERE_Z_THRESHOLD = 5.0

# 심각 이상 채널 최소 개수
MIN_SEVERE_CHANNELS = 2

# 심각 이상 특징 종류 최소 개수
#
# 예:
# RMS + Kurtosis
# RMS + Crest Factor
# Max + Kurtosis
MIN_FEATURE_TYPES = 2

# 몇 segment 연속 만족해야 교체 권고할지
CONSECUTIVE_COUNT = 5


# ============================================================
# 5. 특징 종류
# ============================================================

FEATURE_TYPES = [
    "rms",
    "kurtosis",
    "max",
    "crest_factor",
]


# ============================================================
# 6. 채널 번호 추출
# ============================================================


def get_channel_number(column_name):

    match = re.search(r"channel_(\d+)_", column_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# 7. 특징 종류 추출
# ============================================================


def get_feature_type(column_name):

    if "_crest_factor_" in column_name:
        return "crest_factor"

    elif "_kurtosis_" in column_name:
        return "kurtosis"

    elif "_rms_" in column_name:
        return "rms"

    elif "_max_" in column_name:
        return "max"

    return None


# ============================================================
# 8. 연속 True 시작점 찾기
# ============================================================


def find_consecutive_start(series, consecutive_count):

    rolling_count = series.astype(int).rolling(window=consecutive_count).sum()

    detected = rolling_count[rolling_count >= consecutive_count]

    if len(detected) == 0:
        return None

    end_idx = detected.index[0]

    start_idx = end_idx - consecutive_count + 1

    return start_idx


# ============================================================
# 9. 05번 열화 시작 결과 불러오기
# ============================================================

summary_df = pd.read_csv(SUMMARY_PATH, parse_dates=["degradation_start"])


# ============================================================
# 10. 전체 결과 저장용
# ============================================================

replacement_results = []

replacement_cause_results = []


# ============================================================
# 11. 데이터셋별 분석
# ============================================================

for dataset_name, detail_path in DETAIL_PATHS.items():

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    # ========================================================
    # 데이터 불러오기
    # ========================================================

    df = pd.read_csv(detail_path, parse_dates=["start_time", "end_time"])

    df = df.sort_values("start_time").reset_index(drop=True)

    # ========================================================
    # 12. 열화 시작 시점 가져오기
    # ========================================================

    degradation_row = summary_df[summary_df["dataset"] == dataset_name]

    if degradation_row.empty:

        print("열화 시작 결과 없음")
        continue

    degradation_time = degradation_row["degradation_start"].iloc[0]

    if pd.isna(degradation_time):

        print("열화 시작 시점 없음")
        continue

    print("\n열화 시작:")
    print(degradation_time)

    # ========================================================
    # 13. 필수 특징 Z-score 컬럼 선택
    # ========================================================

    zscore_cols = [
        col
        for col in df.columns
        if (
            col.endswith("_rms_zscore")
            or col.endswith("_kurtosis_zscore")
            or col.endswith("_max_zscore")
            or col.endswith("_crest_factor_zscore")
        )
    ]

    print("\n사용 Z-score 특징 개수:")
    print(len(zscore_cols))

    # ========================================================
    # 14. 심각 이상 특징 flag 생성
    #
    # 각 특징별 Z >= 5
    # ========================================================

    severe_feature_flags = {}

    for col in zscore_cols:

        severe_feature_flags[col] = df[col] >= SEVERE_Z_THRESHOLD

    severe_feature_df = pd.DataFrame(severe_feature_flags)

    # ========================================================
    # 15. 채널별 심각 이상 여부
    #
    # 해당 채널의 4개 특징 중
    # 하나라도 Z >= 5면 심각 이상 채널
    # ========================================================

    channels = sorted(
        set(
            get_channel_number(col)
            for col in zscore_cols
            if get_channel_number(col) is not None
        )
    )

    print("\n채널:")
    print(channels)

    channel_severe_df = pd.DataFrame(index=df.index)

    for channel in channels:

        channel_cols = [
            col for col in zscore_cols if col.startswith(f"channel_{channel}_")
        ]

        channel_severe_df[f"channel_{channel}_severe"] = severe_feature_df[
            channel_cols
        ].any(axis=1)

    # ========================================================
    # 16. 심각 이상 채널 수
    # ========================================================

    severe_channel_count = channel_severe_df.sum(axis=1)

    # ========================================================
    # 17. 특징 종류별 심각 이상 여부
    #
    # 예:
    # RMS 계열 중 하나라도 Z>=5
    # Kurtosis 계열 중 하나라도 Z>=5
    # Max 계열 중 하나라도 Z>=5
    # Crest Factor 계열 중 하나라도 Z>=5
    # ========================================================

    feature_type_df = pd.DataFrame(index=df.index)

    for feature_type in FEATURE_TYPES:

        type_cols = [
            col for col in zscore_cols if get_feature_type(col) == feature_type
        ]

        if len(type_cols) == 0:

            feature_type_df[f"{feature_type}_severe"] = False

        else:

            feature_type_df[f"{feature_type}_severe"] = severe_feature_df[
                type_cols
            ].any(axis=1)

    # ========================================================
    # 18. 심각 이상 특징 종류 개수
    # ========================================================

    severe_feature_type_count = feature_type_df.sum(axis=1)

    # ========================================================
    # 19. 분석용 컬럼 합치기
    # ========================================================

    df = pd.concat(
        [
            df,
            channel_severe_df,
            feature_type_df,
        ],
        axis=1,
    )

    df["severe_channel_count"] = severe_channel_count

    df["severe_feature_type_count"] = severe_feature_type_count

    # ========================================================
    # 20. 교체 후보 판정
    #
    # 조건 1:
    # 심각 이상 채널 >= 2
    #
    # 조건 2:
    # 심각 이상 특징 종류 >= 2
    # ========================================================

    df["replacement_candidate"] = (
        df["severe_channel_count"] >= MIN_SEVERE_CHANNELS
    ) & (df["severe_feature_type_count"] >= MIN_FEATURE_TYPES)

    # ========================================================
    # 21. 열화 시작 이전은 교체 판정 제외
    # ========================================================

    df.loc[df["start_time"] < degradation_time, "replacement_candidate"] = False

    # ========================================================
    # 22. 연속 조건 적용
    # ========================================================

    replacement_idx = find_consecutive_start(
        df["replacement_candidate"], CONSECUTIVE_COUNT
    )

    # ========================================================
    # 23. 교체 권고 시점 계산
    # ========================================================

    dataset_end_time = df["end_time"].iloc[-1]

    if replacement_idx is not None:

        replacement_time = df.loc[replacement_idx, "start_time"]

        replacement_position_pct = replacement_idx / len(df) * 100

        degradation_to_replacement = replacement_time - degradation_time

        replacement_to_dataset_end = dataset_end_time - replacement_time

    else:

        replacement_time = None

        replacement_position_pct = None

        degradation_to_replacement = None

        replacement_to_dataset_end = None

    # ========================================================
    # 24. 결과 출력
    # ========================================================

    print("\n==============================")
    print("교체 권고 결과")
    print("==============================")

    print("열화 시작:")
    print(degradation_time)

    print("\n교체 권고:")
    print(replacement_time)

    print("\n교체 권고 수명 위치(%):")
    print(replacement_position_pct)

    print("\n열화 시작 → 교체 권고:")
    print(degradation_to_replacement)

    print("\n교체 권고 → 데이터 종료:")
    print(replacement_to_dataset_end)

    # ========================================================
    # 25. 교체 권고 시점 원인 분석
    # ========================================================

    abnormal_features = []

    if replacement_idx is not None:

        print("\n==============================")
        print("교체 권고 원인 분석")
        print("==============================")

        print("심각 이상 채널 수:", df.loc[replacement_idx, "severe_channel_count"])

        print(
            "심각 이상 특징 종류 수:",
            df.loc[replacement_idx, "severe_feature_type_count"],
        )

        # ----------------------------------------------------
        # 실제 Z >= 5인 특징 찾기
        # ----------------------------------------------------

        for col in zscore_cols:

            z_value = df.loc[replacement_idx, col]

            if z_value >= SEVERE_Z_THRESHOLD:

                feature_name = col.replace("_zscore", "")

                feature_type = get_feature_type(col)

                channel_number = get_channel_number(col)

                row = {
                    "dataset": dataset_name,
                    "replacement_time": replacement_time,
                    "channel": channel_number,
                    "feature_type": feature_type,
                    "feature": feature_name,
                    "zscore": z_value,
                }

                abnormal_features.append(row)

                replacement_cause_results.append(row)

    abnormal_feature_df = pd.DataFrame(abnormal_features)

    print("\n교체 권고 원인 특징")

    if not abnormal_feature_df.empty:

        print(abnormal_feature_df[["channel", "feature_type", "feature", "zscore"]])

    else:

        print("없음")

    # ========================================================
    # 26. 교체 권고 당시 활성 특징 종류 출력
    # ========================================================

    if replacement_idx is not None:

        active_feature_types = []

        for feature_type in FEATURE_TYPES:

            col = f"{feature_type}_severe"

            if df.loc[replacement_idx, col]:

                active_feature_types.append(feature_type)

        print("\n활성 특징 종류:")
        print(active_feature_types)

    # ========================================================
    # 27. 데이터셋별 저장 폴더
    # ========================================================

    dataset_dir = f"{SAVE_DIR}/" f"{dataset_name}"

    os.makedirs(dataset_dir, exist_ok=True)

    # ========================================================
    # 28. 상세 결과 저장
    # ========================================================

    df.to_csv(f"{dataset_dir}/" f"{dataset_name}_replacement_detail.csv", index=False)

    # ========================================================
    # 29. 원인 특징 저장
    # ========================================================

    if not abnormal_feature_df.empty:

        abnormal_feature_df.to_csv(
            f"{dataset_dir}/" "replacement_features.csv", index=False
        )

    # ========================================================
    # 30. 시각화 1
    # 심각 이상 채널 수
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(df["start_time"], df["severe_channel_count"], label="심각 이상 채널 수")

    plt.axhline(y=MIN_SEVERE_CHANNELS, linestyle="--", label="채널 기준")

    plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")

    if replacement_time is not None:

        plt.axvline(x=replacement_time, linestyle="--", label="교체 권고")

    plt.title(f"{dataset_name} 심각 이상 채널 수")

    plt.xlabel("Time")

    plt.ylabel("Severe Channel Count")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{dataset_dir}/" "severe_channel_count.png", dpi=300, bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ========================================================
    # 31. 시각화 2
    # 심각 이상 특징 종류 수
    # ========================================================

    plt.figure(figsize=(14, 5))

    plt.plot(
        df["start_time"],
        df["severe_feature_type_count"],
        label="심각 이상 특징 종류 수",
    )

    plt.axhline(y=MIN_FEATURE_TYPES, linestyle="--", label="특징 종류 기준")

    plt.axvline(x=degradation_time, linestyle=":", label="열화 시작")

    if replacement_time is not None:

        plt.axvline(x=replacement_time, linestyle="--", label="교체 권고")

    plt.title(f"{dataset_name} 심각 이상 특징 종류")

    plt.xlabel("Time")

    plt.ylabel("Severe Feature Type Count")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{dataset_dir}/" "severe_feature_type_count.png", dpi=300, bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # ========================================================
    # 32. 결과 요약
    # ========================================================

    replacement_results.append(
        {
            "dataset": dataset_name,
            "degradation_start": degradation_time,
            "severe_z_threshold": SEVERE_Z_THRESHOLD,
            "min_severe_channels": MIN_SEVERE_CHANNELS,
            "min_feature_types": MIN_FEATURE_TYPES,
            "consecutive_count": CONSECUTIVE_COUNT,
            "replacement_time": replacement_time,
            "replacement_position_pct": replacement_position_pct,
            "degradation_to_replacement": degradation_to_replacement,
            "dataset_end_time": dataset_end_time,
            "replacement_to_dataset_end": replacement_to_dataset_end,
        }
    )


# ============================================================
# 33. 전체 결과 저장
# ============================================================

result_df = pd.DataFrame(replacement_results)


result_df.to_csv(f"{SAVE_DIR}/" "replacement_summary.csv", index=False)


print("\n" + "=" * 70)
print("최종 교체 권고 결과")
print("=" * 70)

print(result_df)


# ============================================================
# 34. 전체 교체 원인 저장
# ============================================================

cause_df = pd.DataFrame(replacement_cause_results)


cause_df.to_csv(f"{SAVE_DIR}/" "replacement_causes.csv", index=False)


print("\n" + "=" * 70)
print("전체 교체 권고 원인 특징")
print("=" * 70)

print(cause_df)


# ============================================================
# 35. 완료
# ============================================================

print("\n" + "=" * 70)
print("06 교체 권고 분석 완료")
print("=" * 70)
