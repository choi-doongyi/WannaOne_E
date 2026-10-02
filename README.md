# 터보팬 엔진 베어링 진동 기반 열화 탐지 및 교체 시점 분석

## 1. 프로젝트 개요

본 프로젝트는 터보팬 엔진의 다채널 진동 시계열 데이터를 활용하여 **베어링 상태 변화와 열화 시작 시점을 탐지하고, 교체 권고 시점을 도출하는 것**을 목표로 한다.

원시 데이터는 `timestamp`와 8개 진동 채널(`channel_1` ~ `channel_8`)로 구성되어 있으며, 약 1 ms 간격(1 kHz)으로 짧게 측정한 신호가 약 10분 간격으로 반복되는 측정 세션 구조를 가진다.

분석은 다음 흐름으로 구성하였다.

```text
Raw Data
   ↓
DDA / 측정 구조 확인
   ↓
Segment 분리
   ↓
시간·주파수 특징 추출
   ↓
Feature Data 품질 점검
   ↓
EDA / 시계열 변화 분석
   ↓
최종 모델링용 전처리
   ↓
Robust Moving Z-score 열화 탐지
   ↓
교체 권고 기준 적용
   ↓
Isolation Forest 보조 검증
   ↓
상태 변화 시점 비교
   ↓
FFT 기반 추가 분석
```

---

## 2. 분석 목표

- 원시 진동 데이터의 측정 구조와 시간 간격 확인
- 측정 Segment별 시간영역·주파수영역 특징 추출
- 시간에 따른 진동 특징의 변화 및 열화 징후 탐색
- 초기 정상구간을 기준으로 한 **Robust Moving Z-score 열화 시작 탐지**
- 더 강한 이상 상태가 지속되는 시점을 기반으로 **교체 권고 시점 도출**
- **Isolation Forest**를 활용한 다변량 이상 상태 보조 검증
- 열화 시작 → 교체 권고 → 다변량 이상 진행의 시간적 관계 비교
- FFT를 활용한 열화 전·후 주파수 대역 변화 분석

> 실제 고장 시점이나 정비 이력에 대한 Ground Truth가 제공되지 않았으므로, 본 프로젝트의 열화 시작 및 교체 권고 결과는 **운영 기준에 따라 탐지한 상태 변화 후보 시점**으로 해석한다.

---

## 3. 데이터 구조

원본 데이터 파일은 아래와 같이 사용한다.

```text
data/
├─ test_set_1_1ms.csv
├─ test_set_2_1ms.csv
└─ test_set_3_1ms.csv
```

### 원본 컬럼

```text
timestamp
channel_1
channel_2
...
channel_8
```

### 데이터 수집 구조

DDA 결과 대부분의 데이터는 약 1 ms 간격으로 측정되며, 일정 길이의 측정 이후 약 10분 정도의 공백이 존재한다.

따라서 이 공백을 단순 결측으로 처리하지 않고 **측정 세션 사이의 간격**으로 해석하였다.

- Set1: 주로 1,024 samples / segment
- Set2: 주로 512 samples / segment
- Set3: 주로 512 samples / segment
- Sampling Frequency: 1,000 Hz

---

## 4. Segment 특징 추출

1초 이상의 시간 간격이 발생하면 새로운 측정 Segment로 분리하였다.

각 채널에 대해 다음 특징을 추출한다.

### 시간영역 특징

| Feature | 의미 |
|---|---|
| RMS | 전체적인 진동 크기 |
| Kurtosis | 충격성·꼬리 분포 정도 |
| Max | 최대값 |
| Min | 최소값 |
| Crest Factor | Peak / RMS |
| Std | 표준편차 |
| Peak-to-Peak | 최대값 - 최소값 |

### 주파수영역 특징

FFT 수행 전 평균을 제거하여 DC 성분의 영향을 줄였다.

| Feature | 의미 |
|---|---|
| Dominant Frequency | 가장 강한 주파수 |
| Spectral Centroid | 주파수 에너지의 중심 |
| Spectral Energy | 주파수 영역 전체 에너지 |
| Spectral Entropy | 주파수 에너지 분산 정도 |

각 Segment는 최종적으로 **1개의 행**으로 변환되고, 채널별 특징을 별도 컬럼으로 유지한다.

---

## 5. 프로젝트 파일 및 실행 순서

| 순서 | 파일 | 역할 |
|---:|---|---|
| 1 | `DDA.py` | 원시 데이터의 컬럼, 크기, 자료형, 중복, 시간 간격 및 측정 Segment 구조 확인 |
| 2 | `new_set.py` | 원시 1 ms 신호를 Segment로 분리하고 시간·주파수 특징 추출 |
| 3 | `DDA2.py` | 특징 데이터의 중복, 결측, 무한대, Segment 길이, 상수 컬럼 점검 |
| 4 | `EDA.py` | Set1 대표 데이터의 분포, 시간 추세, 상관관계, 초기·중기·후기 특징 변화 탐색 |
| 5 | `03_TimeSeries_EDA.py` | Rolling Mean, 변화점 탐색, 변화시점 집중도 및 Health Indicator 분석 |
| 6 | `04_Final_Preprocessing.py` | 대표 Segment 길이 유지, 결측 처리, 상수 특징 제거 및 모델링용 데이터 생성 |
| 7 | `05_Moving_ZScore_Detection.py` | Robust Moving Z-score 기반 조기 열화 시작 탐지 |
| 8 | `06_Replacement_Criteria.py` | 심각 이상이 지속되는 시점을 이용한 교체 권고 시점 탐지 |
| 9 | `07_Isolation_Forest_Validation.py` | 초기 정상구간으로 Isolation Forest를 학습하고 다변량 이상 시작 시점 탐지 |
| 10 | `08_Final_Timeline_Comparison.py` | 열화 시작·교체 권고·Isolation Forest 이상 시작 시점 통합 비교 |
| 11 | `09_FFT_Analysis.py` | 열화 전·후 FFT 및 주파수 대역 에너지 변화 분석 |

---

## 6. 전처리 흐름

### 6.1 특징 데이터 생성

`new_set.py`

```text
Raw CSV
→ timestamp 정렬
→ 1초 이상 gap 기준 Segment 분리
→ 채널별 시간영역 특징 추출
→ 채널별 주파수영역 특징 추출
→ set1_features.csv
→ set2_features.csv
→ set3_features.csv
```

### 6.2 특징 데이터 품질 점검

`DDA2.py`

확인 항목:

- 중복 행
- 결측치
- `inf`, `-inf`
- Segment별 `n_samples`
- 비정상적으로 짧은 Segment
- 상수 Feature

출력:

```text
data/set1_processed.csv
data/set2_processed.csv
data/set3_processed.csv
```

### 6.3 최종 모델링용 전처리

`04_Final_Preprocessing.py`

대표 Segment 길이와 다른 측정 구간을 제외하여 FFT, RMS 등의 특징을 동일한 측정 조건에서 비교한다.

추가 처리:

- 중복 제거
- `inf`, `-inf` → NaN
- 전체 NaN Feature 제거
- 남은 결측치 Median 대체
- 상수 Feature 제거

출력:

```text
data/model_ready/
├─ set1_model_ready.csv
├─ set2_model_ready.csv
├─ set3_model_ready.csv
├─ set1_feature_list.csv
├─ set2_feature_list.csv
└─ set3_feature_list.csv
```

---

## 7. EDA 및 시계열 분석

### 일반 EDA

`EDA.py`는 Set1을 대표 데이터로 사용하여 다음을 확인한다.

- 특징별 시간 추세
- 분포 및 왜도
- 채널별 특징 상관관계
- 높은 상관관계를 가진 특징 조합
- 전체 수명 구간을 초기 30% / 중기 40% / 후기 30%로 나누어 특징 변화 비교

> 초기·중기·후기는 실제 고장 라벨이 아니라 **전체 관측 기간에서의 상대적인 시간 위치**를 의미한다.

일부 시각화 블록은 필요에 따라 주석을 해제하여 사용할 수 있다.

### 시계열 EDA

`03_TimeSeries_EDA.py`

- 최근 20개 Segment Rolling Mean
- Rolling Mean 변화량 계산
- 특징별 최대 변화점 탐색
- 변화시점 집중 구간 확인
- Health Indicator 생성
- Health Indicator Rolling Mean 확인

Health Indicator는 RMS, Kurtosis, Crest Factor, Peak-to-Peak, Spectral Energy 계열 특징을 표준화한 뒤 평균하여 구성한다.

---

## 8. Robust Moving Z-score 열화 탐지

`05_Moving_ZScore_Detection.py`

초기 30% 데이터를 정상 기준 구간으로 사용한다.

평균과 표준편차보다 극단값에 덜 민감한 Median과 MAD를 사용하여 Robust Z-score를 계산한다.

```text
Robust Z = 0.6745 × (x - Median) / MAD
```

### 열화 시작 판정 기준

```text
Baseline             : 초기 30%
Rolling Window        : 5 segments
Robust Z Threshold    : Z >= 3
Minimum Channels      : 2개 이상
Consecutive Condition : 3 segments 연속
```

한 채널의 다음 특징 중 하나라도 기준을 초과하면 해당 채널을 이상 채널로 판단한다.

- RMS
- Kurtosis
- Max
- Crest Factor

출력:

```text
results/moving_zscore/
├─ moving_zscore_summary.csv
├─ degradation_start_causes.csv
├─ set1/
├─ set2/
└─ set3/
```

---

## 9. 교체 권고 기준

`06_Replacement_Criteria.py`

열화 시작 이후 더 강한 이상 상태가 지속되는 시점을 교체 권고 후보로 정의하였다.

### 교체 권고 조건

```text
분석 시작       : Moving Z-score 열화 시작 이후
Severe Z        : Z >= 5
심각 이상 채널  : 2개 이상
이상 특징 종류  : 2종 이상
지속 조건       : 5 segments 연속
```

사용 특징 종류:

```text
RMS
Kurtosis
Max
Crest Factor
```

따라서 단일 채널의 일시적인 Spike보다 여러 채널·여러 특징에서 동시에 발생하고 지속되는 강한 이상 상태에 더 높은 비중을 둔다.

출력:

```text
results/replacement_criteria/
├─ replacement_summary.csv
├─ replacement_causes.csv
├─ set1/
├─ set2/
└─ set3/
```

---

## 10. Isolation Forest 보조 검증

`07_Isolation_Forest_Validation.py`

Robust Moving Z-score가 통계 기반 열화 탐지라면, Isolation Forest는 여러 특징을 동시에 고려하는 비지도 이상탐지 모델로 사용한다.

### 사용 Feature

- RMS
- Kurtosis
- Max
- Crest Factor

FFT 계열 특징은 Isolation Forest 입력에서 제외한다.

### 모델 설정

```text
Training Data          : 초기 30% 정상구간
n_estimators           : 300
random_state           : 42
Anomaly Threshold      : Baseline anomaly score의 99 percentile
Rolling Window         : 5 segments
Rolling Anomaly Ratio  : 60%
Consecutive Condition  : 3 segments 연속
```

모델은 초기 정상구간만 학습하며 전체 기간에 대해 anomaly score를 계산한다.

본 프로젝트에서는 Isolation Forest를 Moving Z-score의 정답 판정기로 사용하지 않고, **다변량 이상 상태가 본격적으로 형성되는 시점을 확인하기 위한 보조 모델**로 사용한다.

출력:

```text
results/isolation_forest_validation/
├─ isolation_forest_validation_summary.csv
├─ set1/
├─ set2/
└─ set3/
```

---

## 11. 최종 상태 변화 비교

`08_Final_Timeline_Comparison.py`

세 가지 상태 변화 시점을 하나의 Timeline으로 비교한다.

```text
Moving Z-score
→ 초기 열화 감지

Replacement Criteria
→ 유지보수 / 교체 권고 후보

Isolation Forest
→ 다변량 이상 상태 진행 확인
```

비교 항목:

- 열화 시작 시점
- 교체 권고 시점
- Isolation Forest 이상 시작 시점
- 각 시점의 전체 수명 위치(%)
- 열화 → 교체까지의 시간
- 교체 → IF 이상까지의 시간
- 각 시점 → 데이터 종료까지의 시간

출력:

```text
results/final_timeline_comparison/
├─ final_timeline_comparison.csv
├─ set1_timeline_comparison.png
├─ set2_timeline_comparison.png
├─ set3_timeline_comparison.png
└─ all_dataset_timeline_comparison.png
```

---

## 12. FFT 추가 분석

`09_FFT_Analysis.py`

열화 시작 시점을 기준으로 직전 10개 Segment와 직후 10개 Segment를 비교한다.

### 분석 내용

- 채널별 열화 전·후 FFT Spectrum 비교
- 전체 채널 평균 FFT 비교
- Spectral Energy 시간 추세
- 0 ~ 500 Hz 구간을 50 Hz 단위로 분할
- 주파수 Band별 열화 전·후 Energy 변화율 계산
- 가장 크게 증가한 주파수 대역 확인

Sampling Frequency가 1,000 Hz이므로 Nyquist Frequency는 500 Hz이다.

> 축 회전수와 베어링 형상 정보가 제공되지 않았으므로 BPFO, BPFI, BSF, FTF 등 특정 베어링 결함 주파수와 결함 종류를 단정하지 않는다.

출력:

```text
results/fft_analysis/
├─ fft_analysis_summary.csv
├─ all_frequency_band_energy.csv
├─ set1/
├─ set2/
└─ set3/
```

---

## 13. 프로젝트 폴더 구조

```text
project/
│
├─ data/
│  ├─ test_set_1_1ms.csv
│  ├─ test_set_2_1ms.csv
│  ├─ test_set_3_1ms.csv
│  ├─ set1_features.csv
│  ├─ set2_features.csv
│  ├─ set3_features.csv
│  ├─ set1_processed.csv
│  ├─ set2_processed.csv
│  ├─ set3_processed.csv
│  └─ model_ready/
│
├─ results/
│  ├─ eda/
│  ├─ moving_zscore/
│  ├─ replacement_criteria/
│  ├─ isolation_forest_validation/
│  ├─ final_timeline_comparison/
│  └─ fft_analysis/
│
├─ DDA.py
├─ new_set.py
├─ DDA2.py
├─ EDA.py
├─ 03_TimeSeries_EDA.py
├─ 04_Final_Preprocessing.py
├─ 05_Moving_ZScore_Detection.py
├─ 06_Replacement_Criteria.py
├─ 07_Isolation_Forest_Validation.py
├─ 08_Final_Timeline_Comparison.py
├─ 09_FFT_Analysis.py
│
└─ README.md
```

---

## 14. 실행 환경

주요 라이브러리:

```text
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
```

설치 예시:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

---

## 15. 실행 방법

원본 CSV 파일을 `data/` 폴더에 배치한 뒤 아래 순서대로 실행한다.

```bash
python DDA.py
python new_set.py
python DDA2.py
python EDA.py
python 03_TimeSeries_EDA.py
python 04_Final_Preprocessing.py
python 05_Moving_ZScore_Detection.py
python 06_Replacement_Criteria.py
python 07_Isolation_Forest_Validation.py
python 08_Final_Timeline_Comparison.py
python 09_FFT_Analysis.py
```

이전 단계에서 생성된 CSV 및 분석 결과를 다음 단계에서 사용하므로 **실행 순서를 유지해야 한다.**

---

## 16. 분석 방법의 역할 구분

| 방법 | 주요 목적 |
|---|---|
| EDA / Time-Series EDA | 특징 변화와 시간적 패턴 탐색 |
| Robust Moving Z-score | 초기 열화 징후 탐지 |
| Replacement Criteria | 유지보수 개입 후보 시점 도출 |
| Isolation Forest | 다변량 이상 상태 보조 검증 |
| FFT Analysis | 열화 전·후 주파수 구조 변화 해석 |

한 가지 모델로 모든 문제를 해결하기보다 **조기 감지 → 유지보수 의사결정 → 다변량 검증 → 주파수 해석**으로 역할을 나누었다.

---

## 17. 한계 및 개선 방향

### 1. Ground Truth 부재

실제 베어링 고장 시점 또는 정비 이력이 제공되지 않아 Precision, Recall, F1-score와 같은 지도학습 평가 지표로 열화 시점을 검증하기 어렵다.

따라서 현재 결과는 실제 고장 확정값이 아니라 **운영 규칙 기반 상태 변화 후보 시점**이다.

### 2. 초기 30% 정상구간 가정

Moving Z-score와 Isolation Forest 모두 초기 30%를 정상 상태로 가정한다.

초기 구간에 이미 이상 상태가 포함되어 있다면 기준선이 왜곡될 수 있으므로 실제 현장 적용 시 정상 운전 이력 또는 정비 기록과 함께 검증할 필요가 있다.

### 3. Robust Z-score 방향

현재 기준은 `Z >= threshold` 형태이므로 특징이 증가하는 방향의 이상을 중심으로 탐지한다.

감소 방향의 이상까지 분석하려면 특징별 물리적 의미를 검토한 뒤 `|Z|` 또는 특징별 상승·하락 기준을 적용할 수 있다.

### 4. Segment 길이 차이

Set1은 주로 1,024 samples, Set2·3은 주로 512 samples로 구성된다.

최종 모델링 단계에서는 각 데이터셋 내부에서 대표 Segment 길이만 유지하지만, Segment 길이에 영향을 받는 Spectral Energy의 절대 크기를 데이터셋 간 직접 비교할 때는 주의가 필요하다.

### 5. 결함 종류 특정의 한계

회전수 및 베어링 형상 정보가 없기 때문에 FFT에서 특정 주파수 변화가 발견되더라도 BPFO/BPFI 등 실제 결함 종류까지 단정하지 않는다.

---

## 18. 향후 개선

- 실제 정비·고장 이력 확보 후 탐지 결과와 Ground Truth 비교
- FP / FN 기반 임계값 최적화
- 특징별 증가·감소 방향을 반영한 이상 기준 설계
- Segment 길이를 고려한 주파수 Energy 정규화
- 회전수와 베어링 제원 확보 후 결함 주파수 분석
- Raw Segment를 입력으로 하는 1D-CNN 등 딥러닝 모델과의 비교
- 현장 점검 비용과 고장 손실 비용을 반영한 교체 시점 최적화

---

## 19. 데이터 출처

원본 데이터의 공개 링크 또는 배포 출처는 저장소 공개 시 이 항목에 추가한다.

> 대용량 원본 CSV는 Git 저장소에 직접 업로드하지 않고, 다운로드 방법과 배치 경로를 안내하는 방식을 권장한다.

---

## 20. 핵심 요약

본 프로젝트는 단순히 이상 여부를 분류하는 데서 끝나지 않고 다음과 같이 상태 변화를 단계적으로 해석한다.

```text
정상 기준 형성
   ↓
초기 열화 징후 탐지
   ↓
강한 이상 지속
   ↓
교체 권고 후보
   ↓
다변량 이상 진행 확인
   ↓
FFT를 통한 주파수 변화 해석
```

이를 통해 터보팬 엔진 베어링 진동 데이터를 기반으로 **상태 모니터링 → 열화 탐지 → 유지보수 의사결정 → 이상 검증**까지 이어지는 분석 파이프라인을 구성하였다.
