import pandas as pd

df1 = pd.read_csv("data/test_set_1_1ms.csv")
df2 = pd.read_csv("data/test_set_2_1ms.csv")
df3 = pd.read_csv("data/test_set_3_1ms.csv")

# 각각의 데이터셋의 기간이 달라 특징이 다를 수 있기에 합치지않고 따로 계산
data_set = [df1, df2, df3]
for i in data_set:
    # 1. 컬럼명
    print("=" * 10, str(i), "=" * 10)
    print("=" * 10, "column", "=" * 10)
    print(i.columns.tolist())
    print("=" * 10, "shape", "=" * 10)
    print(i.shape)
    print("=" * 10, "info", "=" * 10)
    print(i.info())
    print("=" * 10, "describe", "=" * 10)
    print(i.describe().T)
    print("=" * 10, "duplicated", "=" * 10)
    print(i.duplicated().sum())
    #  각 컬럼 고유값 개수
    print("=" * 10, "nunique", "=" * 10)
    print(i.nunique())

    i["timestamp"] = pd.to_datetime(i["timestamp"])
    print("start :", i["timestamp"].min())
    print("end   :", i["timestamp"].max())

    # 6. 시간 간격 확인
    print("=" * 10, "time interval", "=" * 10)
    print(i["timestamp"].sort_values().diff().value_counts().head(10))

    # 7. 무한값 확인
    numeric_cols = i.select_dtypes(include="number").columns
    print("=" * 10, "infinite values", "=" * 10)
    print(i[numeric_cols].isin([float("inf"), float("-inf")]).sum())
    print()

# 크기
# df1 (2207744, 9)
# df2 (503808, 9)
# df3 (3237888, 9)
# 자료형
# 모두 타임스탬프 빼고 실수형자료
# 이상치가 있는 것으로 추정
# 셋 모두 중복 X
# 대략 10분 정도의 시간간격이 모두 있다! 점검 시간이 존재할 가능성 높다.


for df in data_set:

    df = df.sort_values("timestamp").copy()

    diff = df["timestamp"].diff()

    # 정상 간격 1ms보다 큰 경우
    gaps = diff[diff > pd.Timedelta(milliseconds=1)]

    print(f"\n===== {str(df)} =====")
    print("전체 gap 개수:", len(gaps))

    print("\nGap 크기 상위:")
    print(gaps.value_counts().head(10))

    # gap 때문에 나뉘는 측정 구간 수
    print("측정 segment 수:", len(gaps) + 1)


# 10분 주기로 1ms동안 측정된 사이클 데이터입니다.
# 약 10분 주기로 터보팬 엔진 신호를 짧은 구간 동안 1kHz로 수집한 데이터로 보는게 타당하다.


# 대부분의 데이터는 1ms 간격으로 수집되며
# 일정 개수의 샘플 측정 후 약 10분의 간격을 두고 다음 측정이 수행된다.
# Dataset 1은 주로 1,024 samples, Dataset 2·3은 512 samples 단위의 측정 세션으로 구성된 것으로 추정된다.
# 따라서 약 10분의 시간 간격은 결측보다는 데이터 수집 주기에 따른 측정 간 공백일 가능성이 높다.
