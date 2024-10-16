<div align="center">
  <h2>K-최근접 이웃 알고리즘</h2>
</div>

## K-최근접 이웃 
```
K-Nearest Neighbors, KNN
지도학습 Supervised Learning의 대표적인 분류 Classification 알고리즘 중 하나로, 새로운 데이터를 분류할 때,
가장 가까운 K개의 이웃 데이터를 참조하여 분류를 결정
KNN은 가장 간단하면서도 직관적인 알고리즘 
```
## KNN의 주요 개념
### 1. K
```
- k는 이웃의 개수, 새 데이터를 분류할 때, 해당 데이터와 가장 가까운 k개의 데이터를 찾고,
그 이웃들의 클래스를 참고하여 분류
ex) 예를 들어 k=5로 설정하면 , 새로운 데이터와 가장 가까운 5개의 데이터를 기준으로 해당 데이터 분류
```
### 2. 거리 계산
```
- KNN은 데이터 간의 거리를 계산하여 가장 가까운 이웃을 찾음
보통 유클리드 거리 Euclidean distance를 사용하지만, 맨허트 거리 등 다른 계산 방법도 사용
```
### 3. 분류 방식
```
- KNN은 가장 가까운 이웃들 중에서 다수의 클래스를 선택하는 다수결 방식으로 분류 결정
```
### 4. 훈련 과정 없음
```
- KNN은 별도의 모델 훈련 과정없음
즉 새로운 데이터를 분류할 때마다 훈련 데이터 전체를 참조, 이런 방식 때문에 메모리 기반 알고리즘이라고 불리기도,,
```
### KNN의 장점 => 단순함, 훈련시간 없음, 비선형 분리 
### KNN의 단점 => 느린 예측 속도, 메모리 사용, K값 선택 중요, 특성 스케일링 

# 코드 시작 >>>
### 라이브러리 임포트
```
# ilbrary & data
import numpy as np # 벡터 및 행렬의 연산 처리를 위한 라이브러리
import pandas as pd # 데이터프레임 쉽게 조작하고 분석하는 라이브러리, 특히 데이터셋 읽고 쓰는 데에,,
import matplotlib.pyplot as plt # 데이터의 시각화 라이브러리 -> 데이터 직관적으로 이해 가능 
from sklearn import metrics # 모델 성능 평가 -> 정확도 , 정밀도, 재현율 계산 

# 시험문제 pd.read_csv 뚫고 빈칸 완성하기
dataset = pd.read_csv('../content/sample_data/Data_all/iris.csv') # 경로의 .. 생략해도 가능
```

### search of data 
```
# 시험문제에 나옴

dataset # 엑셀 형식 데이터 변환

# print(dataset) 텍스트 형식 데이터
# 데이터 프레임을 텍스트 형식으로 출력, 각 행과 열이 문자열로 표시, 엑셀 형식 X

# dataset.info() 데이터 정보
# 각 컬럼의 이름, 데이터 타입, 결측치 수, 전체 행 수 등 출력 (데이터의 전반적인 구조 파악)

# dataset.describle() 통계값 요약 자료
# 최소,최대,평균,중앙 등 기본적인 통계 정보 제공, 숫자형 데이터에서만 통계 제공

# dataset.head(5) 첫번째부터 n번째까지의 행만
# 괄호 안에 숫자를 넣으면 그 개수만큼 행 출력

# dataset.tail() 맨 뒤 행부터 n번째까지, n이 없으면 5개

# dataset.columns 참고) row나 rows는 에러 발생

# dataset.shape 행, 열 개수 출력 (data.index는 행 이름 반환)

# dataset.dtype() 컬럼의 데이터 타입 반환

# dataset.isnull().sum() 데이터 전처리 필요성 확인
```

### 데이터에 결측치가 있을 때?
```
1. 결측치 삭제 dataset.dropna() axis=0이 기본, axis=1이 행
2. 결측치 대체
=> 평균값, 중앙값, 최빈값, 고정값, 앞전 값
=> dataset[].mean(), median(), mode()
```
### training과 test dataset 분리
```
# 데이터프레임의 특정 범위의 행과 열을 선택하는 코드
# iloc은 위치기반으로 데이터프레임의 행과 열 선택
# 0: 은 모든 행을 선택 (0에서 끝까지)
# :-1 은 마지막 열을 제외한 모든 열을 선택, -1은 마지막 열 의미, 이를 제외하는 의미
# .values 선택된 데이터프레임은 numpy배열 형식으로 반
X = dataset.iloc[0:,:-1].values # 뒤에서 하나를 뺀 값을 가져와서 x에 저장
# DF의 모든 행 선택, 5번째 열 선택 
y = dataset.iloc[:,4].values # 열은 앞에서 다섯 번째 값만 가져와서 y에 저장

# X.shape X배열의 차원 출력
# X
# y.shape y배열의 차원 출력, 열이 없으면 (행,) 형태로 출력
# y
# print(x)
# print(y)
print(X.shape, y.shape)
```
# Training data : Test data => 80:20
```
# 코드 전체 통으로 시험문제
# train_test_split은 사이킷런에서 제공하는 함수로, 주어진 데이터를 훈련 데이터와 테스트 데이터로 나눔
# 훈련 데이터는 모델을 학습하는데 사용, 테스트 데이터는 학습된 모델의 성능을 평가하는데 사용용
from sklearn.model_selection import train_test_split
# X_train: 훈련용 특성 데이터 X_test: 테스트용 특성 데이터
# y_train: 훈련용 레이블 데이터 y_test: 테스트용 레이블 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) # test dataset 20% 사용

# X_train.shape
# X_test.shape
# y_train.shape y값은 attribute의 값이 없음, 개수만 나옴
# y_test.shape
```
### Data Standardization 표준화: Feature의 범위가 다른 경우에만 사용
```
# 특성 스케일링을 위한 사이킷런 함수
# 데이터를 표준화하여 각 특성(변수)의 평균을 0, 표준편차 1로 맞춤
# 데이터의 범위가 다를 때, 특히 머신러닝 모델에서 특성의 스케일링 차이로 인해 발생할 수 있는 문제 방지
# 특성이 다를 때 표준화를 통해 균형을 나눔 
from sklearn.preprocessing import StandardScaler

# 표준화 작업을 위한 객체(인스턴스) 생성
s = StandardScaler() # 특성 스케일링 scaling, 평균이 0, 표준편차가 1이 되도록 반환

# fit()은 주어진 데이터 X_train에 맞춰 스케일러 학습
# 훈련 데이터의 평균과 표준편차 계산하여 이를 기반으로 표준화 적용할 준비 
s.fit(X_train) # 교재와 다르게 추가한 부분

# fit()으로 학습된 것 사용하여 훈련 데이터를 표준화(스케일링)
# transform()은 데이터를 실제로 변환하는 메소드, X_train은 스케일링된 데이터로 대체
# transform(): 데이터의 특성을 변환하는 메소드, 학습된 평균과 표준편차를 기반으로 데이터 스케일링
X_train = s.transform(X_train) # 훈련 데이터를 스케일링 처리
# 테스트 데이터도 마찬가지로 변환 처리 
X_test = s.transform(X_test) # 테스트 데이터를 스케일링 처리

# X_train
X_test
```

### create model -> KNN
```
# k-최근접 이웃 분류 알고리즘을 제공하는 사이킷런 클래스, 분류 수행 클래스
from sklearn.neighbors import KNeighborsClassifier

#  k값 50 설정
# 홀수로 설정하지 않는 것임 좋음, 동점이 발생하지 않기 때문
# n_neighbors는 참조할 이웃의 수 지정하는 매개변수 
KNN = KNeighborsClassifier(n_neighbors=50) # k의 값은 홀수 : 3,5,7,,,,,49

# .astype(int)는 데이터 타입을 정수형으로 변환하는 함수
#y_train = y_train.astype(int) # fit type error

# KNN모델을 주어진 훈련 데이터로 학습시키는 함수 
KNN.fit(X_train, y_train) # fit
```

### model accuracy
```
from sklearn import metrics

# y_pred는 모델이 예측한 타겟값(레이블) 저장하는 변수
# 모델이 X_test 샘플에 대해 예측한 클래스 저장하게 됨
y_pred = KNN.predict(X_test)

# 타입변환
# y_test = y_test.astype(int)

# metrics.accuracy_score 함수 
# 정확도는 모델이 예측한 값 y_pred와 실제값 y_test를 비교해 정확도 계산
# 모델이 올바르게 예측한 샘플의 비율 나타냄, 전체 예측 중 정확하게 분류된 샘플의 비율
print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred))) # 랜덤으로 k값이 50이기 때문에 정확도는 다름
```

## K값에 따른 정확도 변동
```
- 정확도는 k값에 따라 달라짐
KNN에서 k값이 크면, 모델이 더 많은 이웃을 참고해 일반화된 예측을 수행함, 데이터에 따라 다름
- K값이 너무 작으으면, 과소적합 underfitting의 문제 발생 => 모델이 일반화되어 세부적인 패턴을 놓침
- K값이 너무 크면, 과적합 overfitting이 발생
=> 모델이 학습 데이터에 너무 민감해져 테스트 데이터에 대해 예측 성능 떨어짐 
```
## 최적의 K값 찾기
```
k = 10 # 최적의 k값 찾기

# acc_array는 정확도를 저장할 배열, 크기를 10인 0으로 채워진 numpy배열 생성
# 각 인덱스에 해당하는 k값에 대한 정확도를 저장하는 공간 미리 만들어놓음
# np.zeros(k): 크기 k로 채워진 배열 생성
acc_array = np.zeros(k)
for k in np.arange(1, k+1,2): # 1부터 k+1까지, 2씩 증가하는 값 생성

    # k값 설정해서 분류기 생성, 훈련 데이터와 레이블을 사용해 KNN모델 학습 
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

    # 테스트 데이터에 대한 예측을 수행해, 각 샘플이 어떤 클래스로 분류되는지 반환
    y_pred = KNN.predict(X_test)

    # 실제와 예측을 비교해 정확도 계산
    acc = metrics.accuracy_score(y_test, y_pred)

    # 정확도를 정확도 배열에 저장, k-1은 인덱스 맞추기 위함
    acc_array[k-1] = acc

# acc_array에서 가장 높은 정확도 반환
# np.amax() 배열에서 최댓값 반환 
max_acc = np.amax(acc_array)
acc_list = list(acc_array) # 정확도 배열을 리스트로 변환

# 최고 정확도에 해당하는 인덱스, 즉 최적의 k값을 찾음
# acc_list.index()는 acc_list에서 최고 정확도 max_acc가 있는 위치 반환
# 리스트에서 최댓값 인덱스를 찾음 .index()
k = acc_list.index(max_acc)
print("Accuracy is", metrics.accuracy_score(y_test, y_pred), "for K-value:", k)
```












