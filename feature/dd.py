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
for df in [df1, df2, df3]:
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
# fdgvnmkl,.;
