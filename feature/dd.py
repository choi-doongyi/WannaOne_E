import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # 그래프
import platform
import seaborn as sns  # 씨본 그래프
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # 정확도
from sklearn.dummy import DummyClassifier  # 베이스라인 dummy
from sklearn.metrics import confusion_matrix  # 혼동행렬
from sklearn.metrics import ConfusionMatrixDisplay  # 혼동행렬 heatmap
from sklearn.metrics import precision_score  # 정밀도
from sklearn.metrics import recall_score  # 재현율
from sklearn.metrics import f1_score, precision_score, recall_score  # 조화평균
from sklearn.metrics import classification_report  # 한눈에
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df1 = pd.read_csv("data/test_set_1_1ms.csv")
df2 = pd.read_csv("data/test_set_2_1ms.csv")
df3 = pd.read_csv("data/test_set_3_1ms.csv")

print("=====================================")
# 데이터 크기 확인
print("데이터 1크기:", df1.shape)  # 220만 행
print("데이터 2크기:", df2.shape)  # 50만 행
print("데이터 3크기:", df3.shape)  # 320만 행

print("=====================================")
# 추가 확인: 컬럼명과 앞부분 데이터
print("데이터1 컬럼:", df1.columns.tolist())
print(df1.head(2), df1.tail(2))

print("데이터2 컬럼:", df2.columns.tolist())
print(df2.head(2), df2.tail(2))

print("데이터3 컬럼:", df3.columns.tolist())
print(df3.head(2), df3.tail(2))

print("=====================================")
# info 및 describe 확인
print("데이터1종합정보:", df1.describe(), df1.info())
print("=====================================")
print("데이터2종합정보:", df2.describe(), df2.info())
print("=====================================")
print("데이터3종합정보:", df3.describe(), df3.info())
print("=====================================")
# 결측개수 확인    ## 결측치 없음
print("데이터1 결측개수:", df1.isna().sum())
print("데이터2 결측개수:", df2.isna().sum())
print("데이터3 결측개수:", df3.isna().sum())
# datetime 인덱스 정렬
for df in [df1, df2, df3]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
df1 = df1.set_index("timestamp").sort_index()
df2 = df2.set_index("timestamp").sort_index()
df3 = df3.set_index("timestamp").sort_index()
#  파일 단위로 RMS·첨도·최댓값·파고율을 계산해 시간 순으로 이어 붙이기
channels = [
    "channel_1",
    "channel_2",
    "channel_3",
    "channel_4",
    "channel_5",
    "channel_6",
    "channel_7",
    "channel_8",
]

data_set = [df1, df2, df3]
for df in data_set:
    # RMS
    df["RMS"] = np.sqrt((df[channels] ** 2).mean(axis=1))

    # 첨도
    df["Kurtosis"] = df[channels].kurtosis(axis=1)

    # 최댓값
    df["Max"] = df[channels].max(axis=1)

    # 파고율 = 최댓값 / RMS
    df["Crest_Factor"] = df["Max"] / df["RMS"]

print("각 데이터에 RMS, 첨도, 최댓값, 파고율 컬럼이 들어갔는지 확인:")
print(df1.head(3))
print(df2.head(3))
print(df3.head(3))
### 각 생성된 컬럼들을 시계열 분석
features = ["RMS", "Kurtosis", "Max", "Crest_Factor"]


### 시계열 그래프
### 한눈에 3개 비교
def plot_three_rows(df1, df2, df3):
    data_list = [
        (df1, "data1", "blue"),
        (df2, "data2", "orange"),
        (df3, "data3", "green"),
    ]

    for col in features:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        for ax, (df, name, color) in zip(axes, data_list):
            df_ = df.copy()

            ax.plot(df_.index, df_[col], color=color)
            ax.set_title(f"{name} - {col}")
            ax.set_xlabel("Time")
            ax.set_ylabel(col)
            ax.grid(True)

        plt.tight_layout()
        plt.show()


plot_three_rows(df1, df2, df3)
### 3 데이터 모두 측정 종료시점에 max치가 치솟는 것으로 보아
### 고장시점이라고 예상할 수 있음


### 새 컬럼들에 대해서 z_score 이상치 탐색해보기
### 이상치 수
def detect_outliers_zscore(df, name, features, threshold=3):
    df_z = df.copy()

    print("=====================================")
    print(f"{name} Z-score 이상치 탐색")

    for col in features:
        mean = df_z[col].mean()
        std = df_z[col].std()

        z_col = f"{col}_zscore"
        outlier_col = f"{col}_outlier"

        df_z[z_col] = (df_z[col] - mean) / std
        df_z[outlier_col] = df_z[z_col].abs() > threshold

        outlier_count = df_z[outlier_col].sum()

        print(f"{col} 이상치 개수: {outlier_count}")

    return df_z


df1_z = detect_outliers_zscore(df1, "data1", features)
df2_z = detect_outliers_zscore(df2, "data2", features)
df3_z = detect_outliers_zscore(df3, "data3", features)


### 이상치 그래프 (3기준)
def plot_zscore_outliers(df, name, features, threshold=3):
    for col in features:
        z_col = f"{col}_zscore"

        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[z_col], label="Z-score")
        plt.axhline(threshold, color="red", linestyle="--", label=f"+{threshold}")
        plt.axhline(-threshold, color="red", linestyle="--", label=f"-{threshold}")
        plt.title(f"{name} - {col} Z-score")
        plt.xlabel("Index")
        plt.ylabel("Z-score")
        plt.legend()
        plt.grid(True)
        plt.show()


# plot_zscore_outliers(df1_z, "data1 z", features)       ### threshold=3인 z_score 그래프를 보여줌

# plot_zscore_outliers(df2_z, "data2 z", features)

plot_zscore_outliers(df3_z, "data3 z", features)

### 컬럼의 수가 많아 z_score만으로 이상을 탐지하는데 한계가 있다고 판단하여
### isolationForest를 사용하여 모델학습을 하려함

X1 = df1[["RMS", "Kurtosis", "Max", "Crest_Factor"]]
X2 = df2[["RMS", "Kurtosis", "Max", "Crest_Factor"]]
X3 = df3[["RMS", "Kurtosis", "Max", "Crest_Factor"]]
# 1%를 이상치로 판단
model_iso1 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model_iso2 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model_iso3 = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

pred1 = model_iso1.fit_predict(X1)
pred2 = model_iso2.fit_predict(X2)
pred3 = model_iso3.fit_predict(X3)

df1["iso_pred"] = pred1
df2["iso_pred"] = pred2
df3["iso_pred"] = pred3

model_iso1.fit(X1)
pred2 = model_iso1.predict(X2)
# df1을 정상 기준으로 두고 df2를 판단
pred2 = model_iso2.fit_predict(X2)
# df2 내부 분포 기준으로 df2의 이상치를 판단
