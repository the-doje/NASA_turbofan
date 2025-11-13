# 🚀 터빈 엔진 RUL 예측 프로젝트

## 📋 프로젝트 개요

항공기 터빈 엔진의 **잔여 수명(RUL, Remaining Useful Life)** 을 예측하는 머신러닝 프로젝트입니다. NASA의 C-MAPSS 데이터셋을 사용하여 예지 정비(Predictive Maintenance)를 위한 최적 모델과 전처리 전략을 체계적으로 탐색합니다.

---

## 🎯 프로젝트 목표

1. **데이터 복잡도에 따른 최적 모델 선택**
2. **효과적인 전처리 전략 발견**
3. **최적 스케일러 선택**
4. **하이퍼파라미터 최적화를 통한 성능 극대화**
5. **일반화 가능한 실험 방법론 확립**

---

## 📊 데이터셋

### **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

| Dataset | 운영 조건 | 고장 모드 | 학습 엔진 | 테스트 엔진 | 복잡도 |
|---------|----------|----------|-----------|-------------|--------|
| **FD001** | 1개 | 1개 | 100대 | 100대 | ⭐ 단순 |
| **FD004** | 6개 | 2개 | 249대 | 248대 | ⭐⭐⭐ 복잡 |

**센서 데이터:** 21개 센서 × 시간 시계열 데이터

**데이터 분할:**
- Training Set: 모델 학습 (80%) + 검증 (20%)
- Test Set: 최종 평가 (1회만 사용)

---

## 🔬 실험 구조: 4-Branch 접근법

```
Branch 1: Model Selection (모델 선택)
    ↓
Branch 2: Preprocessing (전처리 최적화)
    ↓
Branch 3: Scaler (스케일러 최적화)
    ↓
Branch 4: Hyperparameter Tuning (하이퍼파라미터 튜닝)
    ↓
[Branch 5: Ensemble] (예정)
```

---

## 📁 프로젝트 구조

```
├── branch1_FD001.ipynb          # Branch 1: FD001 모델 비교 (17 cells)
├── branch1_FD004.ipynb          # Branch 1: FD004 모델 비교 (17 cells)
├── branch2_FD001.ipynb          # Branch 2: FD001 전처리 실험 (26 cells)
├── branch2_FD004.ipynb          # Branch 2: FD004 전처리 실험 (26 cells)
├── branch3_FD001.ipynb          # Branch 3: FD001 스케일러 비교 (24 cells)
├── branch3_FD004.ipynb          # Branch 3: FD004 스케일러 비교 (24 cells)
├── branch4_FD001.ipynb          # Branch 4: FD001 하이퍼파라미터 튜닝 (23 cells)
├── branch4_FD004.ipynb          # Branch 4: FD004 하이퍼파라미터 튜닝 (23 cells)
└── README.md                    # 프로젝트 문서
```

---

## 🔵 Branch 1: Model Selection

### **목표**
> "어떤 모델이 이 문제에 적합한가?"

### **실험 모델**

#### **1-A: Tabular (표 형식 처리)**
- **XGBoost** (Gradient Boosting)
- **RandomForest** (Bagging)

#### **1-B: Fixed Window (고정 윈도우 시계열)**
- **LSTM-30** (window_size=30)
- **LSTM-50** (window_size=50)

#### **1-C: Variable Window (가변 윈도우 시계열)**
- **TCN** (Temporal Convolutional Network)

### **주요 결과**

| Dataset | Best Model | Test RMSE | Test R² | 선택 이유 |
|---------|-----------|-----------|---------|----------|
| **FD001** | LSTM-50 | 37.23 | 0.49 | 시계열 패턴 명확, 장기 의존성 학습 |
| **FD004** | XGBoost | 72.29 | 0.38 | 복잡한 비선형 관계, 6가지 운영 조건 |

### **핵심 발견 ⭐**

```
✅ 데이터 복잡도가 모델 선택을 결정

FD001 (단순) → LSTM 최적
  - 1개 운영 조건
  - 시계열 패턴 명확
  - 순환 구조가 장기 의존성 학습

FD004 (복잡) → XGBoost 최적
  - 6가지 운영 조건
  - 복잡한 비선형 상호작용
  - Tree 구조가 조건별 분기 학습

💡 인사이트: 
"복잡도 낮은 시계열 → LSTM"
"복잡도 높은 시계열 → Tree 모델"
```

### **노트북 파일**
- `branch1_FD001.ipynb` - FD001 모델 비교 (17 cells)
- `branch1_FD004.ipynb` - FD004 모델 비교 (17 cells)

---

## 🔵 Branch 2: Preprocessing

### **목표**
> "어떤 전처리가 효과적인가?"

### **사전 분석**

#### **1. Sensor Removal Analysis**
- 상관계수 낮은 센서: sensor_1, 5, 16, 18, 19
- 이상치 많은 센서: sensor_8, 14, 15, 18, 19
- **제거 대상:** 8개 센서

#### **2. RUL Clipping Analysis**
- 초기 정상 구간 (RUL > 125): 노이즈 다수, Target 범위 과도
- **최적 임계값:** RUL ≤ 125

### **실험 조건**

| Condition | 설명 |
|-----------|------|
| **Baseline** | 전처리 없음 (원본 데이터) |
| **RUL clipping** | RUL ≤ 125로 제한 |
| **Sensor removal** | 8개 센서 제거 |
| **Both** | RUL clipping + Sensor removal |

### **주요 결과**

#### **FD001 (LSTM-50)**

| Condition | Train RMSE | Test RMSE | Test MAE | 개선율 |
|-----------|------------|-----------|----------|--------|
| Baseline | 24.85 | 36.36 | 25.39 | - |
| **RUL clipping** | 8.60 | **15.62** | 12.70 | **57.04%** ⭐ |
| Sensor removal | 22.74 | 38.34 | 27.45 | -5.44% |
| Both | 11.62 | 16.76 | 14.42 | 53.90% |

#### **FD004 (XGBoost)**

| Condition | Train RMSE | Test RMSE | Test MAE | 개선율 |
|-----------|------------|-----------|----------|--------|
| Baseline | 46.77 | 72.34 | 53.41 | - |
| **RUL clipping** | 14.45 | **16.24** | 10.41 | **77.54%** ⭐ |
| Sensor removal | 47.67 | 72.91 | 53.83 | -0.79% |
| Both | 14.86 | 16.49 | 10.61 | 77.20% |

### **핵심 발견 ⭐**

```
✅ RUL Clipping이 핵심!
  - FD001: 57% 개선
  - FD004: 77% 개선
  - Target 범위 축소 + 초기 노이즈 제거

❌ Sensor Removal 단독은 비효과적
  - 정보 손실 > 노이즈 제거
  - 오히려 성능 악화

💡 인사이트:
"올바른 전처리 하나 > 여러 복잡한 전처리"
"RUL 전처리가 압도적 중요성"
"사전 EDA를 통한 문제 이해가 핵심"
```

### **노트북 파일**
- `branch2_FD001.ipynb` - FD001 전처리 실험 (26 cells, LSTM-50 기반)
- `branch2_FD004.ipynb` - FD004 전처리 실험 (26 cells, XGBoost 기반)

---

## 🔵 Branch 3: Scaler

### **목표**
> "어떤 스케일러가 적합한가?"

### **전제 조건**
- **전처리:** RUL Clipping만 적용
- **모델:** LSTM (window_size=50)
- **이유:** Tree 모델은 스케일링 불필요

### **실험 스케일러**

| Scaler | 특징 | 이상치 대응 |
|--------|------|------------|
| **StandardScaler** | 평균 0, 표준편차 1 | 약함 |
| **MinMaxScaler** | [0, 1] 범위 | 매우 약함 |
| **RobustScaler** | 중앙값, IQR 기반 | **강함** ⭐ |
| **MaxAbsScaler** | [-1, 1] 범위 | 중간 |

### **주요 결과**

#### **FD001**

| Scaler | Train RMSE | Test RMSE | Test MAE | Test R² | 개선율 |
|--------|------------|-----------|----------|---------|--------|
| **Standard** | 7.66 | **15.70** | 11.73 | 0.749 | **0.00%** ⭐ |
| MinMax | 41.43 | 41.06 | 37.38 | -0.716 | **-161.56%** ❌ |
| Robust | 7.80 | 16.23 | 12.34 | 0.732 | -3.38% |

#### **FD004**

| Scaler | Train RMSE | Test RMSE | Test MAE | Test R² | 개선율 |
|--------|------------|-----------|----------|---------|--------|
| Standard | 41.92 | 37.26 | 34.99 | -0.679 | 0.00% |
| MinMax | 41.92 | 37.19 | 34.92 | -0.673 | 0.19% |
| **Robust** | 15.45 | **19.48** | 15.80 | 0.541 | **47.70%** ⭐ |

### **핵심 발견 ⭐**

```
⚠️ 데이터 특성에 따라 최적 스케일러가 다름!

✅ FD001 (단순) → StandardScaler 최적
  - 균일한 센서 분포
  - 정규분포 가정 적합
  - 이상치 적음

✅ FD004 (복잡) → RobustScaler 압도적
  - 6가지 운영 조건
  - 이상치 많음
  - 47% 압도적 개선

❌ MinMaxScaler는 LSTM에 치명적
  - FD001: -161% (완전 붕괴)
  - Gradient 학습 방해
  - [0,1] 범위와 tanh 불일치

💡 인사이트:
"복잡도 ↑ → 스케일러 영향 ↑↑"
"FD001: 3% 차이, FD004: 95% 차이"
"RobustScaler ≠ 만능 (데이터별 테스트 필수)"
```

### **노트북 파일**
- `branch3_FD001.ipynb` - FD001 스케일러 비교 (24 cells)
- `branch3_FD004.ipynb` - FD004 스케일러 비교 (24 cells)

---

## 🔵 Branch 4: Hyperparameter Tuning

### **목표**
> "최적의 파라미터 조합은?"

### **고정 조건**
- **전처리:** RUL Clipping
- **스케일러:** 
  - FD001: StandardScaler (Branch 3 최적)
  - FD004: RobustScaler (Branch 3 최적)
- **모델:**
  - FD001: LSTM-50
  - FD004: XGBoost, LightGBM

### **튜닝 전략**

```
🔍 2단계 접근법:

1단계: Random Search (넓은 범위 탐색)
  - FD001: 10회 시도
  - FD004: 15회 시도
  - 목적: 유망 영역 발견

2단계: Manual Grid Search (집중 탐색)
  - Best 주변 세밀 탐색
  - 목적: 정확한 최적값 발견

💡 왜 2단계?
  - Random만: 정확도 ↓
  - Grid만: 계산 비용 ↑↑
  - 2단계: 효율 + 정확 ✅
```

### **주요 결과**

#### **FD001 (LSTM)**

**최적 하이퍼파라미터:**
```python
Units: [150, 75, 35]      # 큰 네트워크 (1.5배)
Dropout: 0.3              # 중간 정규화
Learning Rate: 0.005      # 중간 학습률
Batch Size: 64            # 작은 배치
Optimizer: RMSprop        # Adam 아님! ⭐
```

| 지표 | 튜닝 전 (Branch 3) | 튜닝 후 (Branch 4) | 개선율 |
|------|-------------------|-------------------|--------|
| **Test RMSE** | 15.70 | **13.92** | **+1.13%** |
| **Test MAE** | 11.73 | **10.04** | **+14.41%** |

#### **FD004 (XGBoost vs LightGBM)**

**XGBoost 최적 파라미터:**
```python
n_estimators: 500         # 많은 트리
max_depth: 6              # 중간 깊이
learning_rate: 0.025      # 작은 학습률
subsample: 0.7            # 강한 정규화
colsample_bytree: 0.7     # 강한 정규화
min_child_weight: 1       # 유연한 분할
gamma: 0.2                # 높은 정규화
```

| Model | Test RMSE | Test MAE | Test R² | 선택 |
|-------|-----------|----------|---------|------|
| XGBoost | 16.09 | 10.50 | 0.601 | - |
| **LightGBM** | **16.03** | **10.35** | **0.604** | ⭐ |

**개선율 (Branch 2 → Branch 4):**
- Test RMSE: 16.49 → 16.03
- **2.79% 개선** ⭐

### **핵심 발견 ⭐**

```
✅ 튜닝 전략이 데이터 특성 반영

FD001: 용량 확대 + 중간 정규화
  - 큰 네트워크 (1.5배)
  - Dropout 0.3
  - RMSprop > Adam ⭐

FD004: 안정성 우선 + 강한 정규화
  - 많은 트리 (500)
  - Subsample 0.7, Gamma 0.2
  - LightGBM 근소 우위

✅ 개선 패턴의 차이

FD001:
  - RMSE: 1.13% 개선
  - MAE: 14.41% 개선 (!)
  → 큰 오차(이상치) 개선에 집중

FD004:
  - RMSE: 2.79% 개선
  - MAE: 비슷한 비율
  → 전반적 정확도 향상

✅ RMSprop의 재발견
  - 통념: Adam이 거의 항상 우수
  - 실제: RMSprop가 터빈 데이터에 더 적합
  - 이유: Momentum 없이 센서별 독립 학습

💡 인사이트:
"한계 수익 체감: 전처리 50-77% > 스케일러 10% > 튜닝 1-3%"
"데이터 복잡도가 튜닝 철학을 결정"
"통념을 깬 발견의 가치"
```

### **노트북 파일**
- `branch4_FD001.ipynb` - FD001 하이퍼파라미터 튜닝 (23 cells)
- `branch4_FD004.ipynb` - FD004 하이퍼파라미터 튜닝 (23 cells)

---

## 📈 최종 성능 요약

### **FD001 (단순 데이터)**

| Stage | Model | Method | Test RMSE | 누적 개선 |
|-------|-------|--------|-----------|----------|
| Branch 1 | LSTM-50 | Baseline | 37.23 | - |
| Branch 2 | LSTM-50 | + RUL Clipping | 15.62 | 58.0% ⬇️ |
| Branch 3 | LSTM-50 | + StandardScaler | 15.70 | 57.8% ⬇️ |
| **Branch 4** | **LSTM-50** | **+ HP Tuning** | **13.92** | **62.6% ⬇️** |

### **FD004 (복잡 데이터)**

| Stage | Model | Method | Test RMSE | 누적 개선 |
|-------|-------|--------|-----------|----------|
| Branch 1 | XGBoost | Baseline | 72.29 | - |
| Branch 2 | XGBoost | + RUL Clipping | 16.49 | 77.2% ⬇️ |
| Branch 3 | LSTM-50 | + RobustScaler | 19.48 | 73.1% ⬇️ |
| **Branch 4** | **LightGBM** | **+ Robust + HP Tuning** | **16.03** | **77.8% ⬇️** |

---

## 💡 핵심 인사이트

### **1. 데이터 복잡도가 모든 것을 결정한다 ⭐⭐⭐**

```
단순 데이터 (FD001):
✓ 모델: LSTM (시계열 패턴 명확)
✓ 전처리: RUL Clipping (57% 개선)
✓ 스케일러: StandardScaler (균일 분포)
✓ 튜닝: 큰 네트워크 + 중간 정규화

복잡 데이터 (FD004):
✓ 모델: Tree (비선형성 강함)
✓ 전처리: RUL Clipping (77% 개선)
✓ 스케일러: RobustScaler (이상치 대응)
✓ 튜닝: 많은 트리 + 강한 정규화
```

### **2. 전처리가 가장 중요하다 ⭐⭐⭐**

```
개선 기여도:
- 전처리 (Branch 2):  50-77% 🚀🚀🚀
- 스케일러 (Branch 3): 5-10% 📈
- 튜닝 (Branch 4):     1-3% 📊

→ 초기 단계가 압도적으로 중요!
→ "올바른 전처리 하나 > 여러 복잡한 기법"
```

### **3. "적을수록 좋다"는 항상 맞지 않다**

```
FD001: 큰 네트워크가 더 우수
- Units [150, 75, 35] (1.5배 크기)
- Dropout으로 과적합 방지 가능
- 시계열 패턴은 복잡 → 용량 필요
```

### **4. 이상치에 강건한 방법 필수**

```
MinMaxScaler vs RobustScaler:
- MinMaxScaler: -161% (FD001, 완전 실패)
- RobustScaler: +47% (FD004, 압도적)

→ 터빈 고장 직전 극단값 대응 중요
```

### **5. 통념을 깬 발견의 가치**

```
RMSprop > Adam (FD001)
- 일반: Adam이 거의 항상 우수
- 실제: RMSprop가 터빈 데이터에 더 적합
- 이유: 센서별 스케일 차이 → Momentum 방해

→ "반드시 실험으로 검증하라"
```

### **6. 한계 수익 체감 법칙**

```
Branch 2: 50-77% 개선 (20-30배)
Branch 3: 5-10% 개선 (2-5배)
Branch 4: 1-3% 개선 (1배)

→ 각 단계마다 개선폭 1/5~1/10 감소
→ 실무: 전처리 80%, 튜닝 20% 노력 투자
```

---

## 🛠️ 기술 스택

### **핵심 라이브러리**

```python
# 데이터 처리
pandas==2.0.3
numpy==1.24.3

# 머신러닝
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.0.0

# 딥러닝
tensorflow==2.13.0
keras==2.13.1

# 시각화
matplotlib==3.7.2
seaborn==0.12.2
```

### **주요 모델**

- **LSTM** (Long Short-Term Memory) - 순환 신경망
- **XGBoost** (Extreme Gradient Boosting) - 그래디언트 부스팅
- **LightGBM** (Light Gradient Boosting Machine) - 경량 부스팅
- **RandomForest** (Random Forest Regressor) - 랜덤 포레스트
- **TCN** (Temporal Convolutional Network) - 시간적 합성곱

---

## 🚀 실행 방법

### **1. 환경 설정**

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 라이브러리 설치
pip install pandas numpy scikit-learn
pip install xgboost lightgbm
pip install tensorflow matplotlib seaborn
```

### **2. 데이터 준비**

```python
# 데이터 경로 설정
DATA_PATH = '/path/to/your/data'

# 필요 파일:
# - FD001_train_df.csv
# - FD001_test_df.csv
# - FD004_train_df.csv
# - FD004_test_df.csv
```

### **3. 노트북 실행**

```bash
# Jupyter Notebook 실행
jupyter notebook

# 순서대로 실행:
# 1. branch1_FD001.ipynb, branch1_FD004.ipynb
# 2. branch2_FD001.ipynb, branch2_FD004.ipynb
# 3. branch3_FD001.ipynb, branch3_FD004.ipynb
# 4. branch4_FD001.ipynb, branch4_FD004.ipynb
```

---

## 🎓 배운 점 (Lessons Learned)

### **1. 방법론**
- ✅ 체계적인 Ablation Study의 중요성
- ✅ Baseline 설정과 단계적 개선의 효과
- ✅ 데이터 특성 이해가 모든 결정의 기반
- ✅ 통념을 의심하고 실험으로 검증

### **2. 기술적**
- ✅ 복잡도에 따른 모델 선택 기준
- ✅ 전처리의 압도적 중요성
- ✅ 이상치 대응의 필수성
- ✅ 적응적 학습률의 가치

### **3. 실무적**
- ✅ 단순한 방법이 복잡한 방법보다 나을 수 있음
- ✅ EDA를 통한 문제 이해가 최우선
- ✅ 과도한 Feature Engineering보다 올바른 전처리
- ✅ 한계 수익 체감 고려한 자원 배분

---

## 📌 다음 단계 (Next Steps)

### **Branch 5: Ensemble (예정)**

```python
계획:
1. LSTM (FD001) + LightGBM (FD004) 조합
2. Simple Averaging / Weighted Averaging
3. Stacking (메타 모델)

기대 효과:
- 추가 2-5% 성능 개선
- 모델 간 오류 상호 보완
- 최종 RMSE 목표: FD001 < 13, FD004 < 15
```

### **추가 실험 아이디어**

- [ ] FD002, FD003 데이터셋 적용
- [ ] Attention Mechanism 도입
- [ ] Transfer Learning (데이터셋 간)
- [ ] 실시간 예측 시스템 구축
- [ ] 설명 가능한 AI (SHAP, LIME)

---

## 📚 참고 자료

### **데이터셋**
- [NASA C-MAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set"

### **논문**
- Saxena, A., et al. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation"
- Heimes, F. O. (2008). "Recurrent neural networks for remaining useful life estimation"
- Zheng, S., et al. (2017). "Long Short-Term Memory Network for Remaining Useful Life estimation"

### **관련 프로젝트**
- [PHM Society Data Challenge](https://www.phmsociety.org/competition/phm/08)
- [Kaggle: Predictive Maintenance](https://www.kaggle.com/competitions/predictive-maintenance)

---

## 📊 실험 결과 요약

### **최종 성능 비교표**

| Dataset | Baseline | 최종 모델 | RMSE | 개선율 | 주요 기법 |
|---------|----------|----------|------|--------|----------|
| **FD001** | LSTM-50 | LSTM-50 Tuned | 13.92 | 62.6% | RUL Clipping + StandardScaler + RMSprop |
| **FD004** | XGBoost | LightGBM Tuned | 16.03 | 77.8% | RUL Clipping + RobustScaler + 강한 정규화 |

---

## 🌟 결론

### **핵심 메시지:**

> **"데이터 복잡도가 모든 전략을 결정한다"**

두 데이터셋 모두 하이퍼파라미터 튜닝의 개선폭이 작았습니다(1-3%). 이를 통해 **전처리가 성능 최적화의 핵심**임을 확인했으며, 데이터 특성에 맞는 체계적 접근의 중요성을 증명했습니다.

향후 앙상블을 통한 추가 개선 가능성을 탐색하고자 합니다.

---

**Last Updated:** 2025-01-10

**Version:** 1.0.0 (Branch 1-4 Complete)

---

<div align="center">

### 🚀 **"전처리 > 스케일러 > 튜닝"** 🚀

**Made with ❤️ for Predictive Maintenance**

</div>
