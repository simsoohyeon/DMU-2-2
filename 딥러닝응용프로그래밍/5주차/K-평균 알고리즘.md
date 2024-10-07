![image](https://github.com/user-attachments/assets/268b1c23-b2c0-49f2-b1d1-f7207344cc79)<div align="center">
<h2>🟠 Colab</h2>
</div>

## 🟣 K-Means (비지도학습)
### 데이터를 미리 정의된 개수의 군집(Cluster)로 나누는 알고리즘 
```
'비지도': 데이터에 레이블이 없다는 뜻,
K-Means는 데이터를 그룹화할 때 사전에 어떤 라벨 정보 없이, 데이터의 특성만을 바탕으로 군집을 찾아냄
```

### K-Means 알고리즘의 동작 과정
#### 1. 군집 수 K 설정
```
먼저 몇 개의 군집으로 데이터를 나눌지, K값을 설정, 사전에 결정해야 함
```
#### 2. 초기 중심점 설정
```
K개의 군집에 대한 초기 중심(centroid)을 임의로 선택함,
이 중심점은 군집의 중심을 나타내며, 나중에 데이터에 따라 업데이트 
```
#### 3. 각 데이터 포인트를 가장 가까운 중심에 할당
```
모든 데이터를 각 군집 중심점과 비교하여 가장 가까운 중심점에 해당하는 군집으로 할당
유클리드 거리 (Euclidean Distance)를 사용하여 각 데이터 포인트와 중심점 사이의 거리 계산
```
#### 4. 중심점 재계산
```
모든 데이터가 군집에 할당되면, 각 군집의 새로운 중심점 계산
새로운 중심점은 해당 군집 내 데이터 포인트들의 평균으로 계산
```
#### 5. 군집 할당과 중심점 재계산 반복
```
데이터 포인트의 할당과 중심점 재계산 단계를 반복, 더 이상 중심점이 변하지 않거나
데이터 포인트의 할당이 더 이상 바뀌지 않을 때 알고리즘은 수렴하여 종료
```
## 🟣 K-Means의 특징
```
⚫ 비지도학습: 데이터에 라벨이 없을 때 데이터를 분류하는데 유용
⚫ 빠르고 간단함: 알고리즘이 비교적 단순하고 빠르지만, K 값을 미리 설정해야 한다는 단점
⚫ 구형 군집: K-Means는 유클리드 거리 기반이므로 군집이 구형에 가까운 형태로 나뉠 때 성능이 좋음
복잡한 형태의 군집에서는 성능이 떨어질 수 있음
⚫ 군집 수에 의존적: K값을 적절하게 설정하지 않으면, 군집이 적절히 나누어지지 않을 수 있음
엘보우 방법(Elbow Method)나 실루엣분석(Silhouette Analysis)등 통해 K값 선택
```

## 🟣 다양한 라이브러리와 모듈 임포트
#### ◼️ import numpy as np
```
=> 수치 연산을 위한 파이썬 라이브러리, 배열 및 행렬 연산, 수학적 연산을 처리할 때 사용
```
#### ◼️ import pandas as pd
```
=> 데이터 분석을 위한 라이브러리, 데이터 프레임 형태로 데이터를 다룸
데이터 로딩, 전처리, 탐색 등에 자주 사용
```
#### ◼️ import matplotlib.pyplot as plt
```
=> 데이터 시각화를 위한 라이 브러리, pyplot은 그래프를 그리기 위한 모듈
데이터 분포나 결과를 시각화할 때 사용
```
#### ◼️ import tensorflow as tf
```
=> Tensorflow는 머신러닝과 딥러닝을 위한 프레임 워크
딥러닝 관련 작업을 위한 준비, 모델 학습에 사용
```
#### ◼️ from sklearn import svm, datasets
```
=> scikit-learn은 머신러닝을 위한 파이썬 라이브러리, svm은 서포트 벡터 머신 모델을 위한 모듈,
datasets은 기존 제공하는 데이터셋을 로드하기 위한 모듈 
```
#### ◼️ from sklearn import metrics 
```
=> metrics 모듈은 모델의 성능을 평가하는 다양한 지표 제공, 정확도, 정밀도, 재현율 계산
```
#### ◼️ from sklearn.model.selection import train_test_split
```
=> train_test_split 함수는 데이터를 학습용과 테스트용으로 나누기 위한 도구,
데이터를 무작위로 분해해 학습 및 검증 과정을 진행할 수 있도록 함
```
#### ◼️ from sklearn.svm import SVC
```
=> SVC는 서포트 벡터 분류(Support Vector Classifier) 모델, SVM을 사용해 분류 작업 수행
SVM 알고리즘을 기반으로 분류 모델을 정의할 때 사용
```
#### ◼️ from sklearn.preprocessing import MinMaxScaler
```
=> MinMaxScaler는 데이터를 정규화하는 도구, 데이터를 0과 1사이 값으로 반환해 모델이 학습하기 쉽게
데이터 스케일링을 통해 학습 성능을 향상시키기 위해 사용
```
#### ◼️ from sklearn.cluster import KMeans
```
=> KMeans는 k-평균 군집화 알고리즘을 제공하는 클래스, 비지도학습으로 데이터 포인트를 K개의 군집으로 나
```
## 🟣 데이터 로드
```
from google.colab import files
file_uploaded=files.upload()

data = pd.read_csv('sales data.csv') # CSV 파일 읽어서 DF 형식으로 변환
data.head()
```
#### ◼️ from google.colab import files
```
=> Google Colab에서 제공하는 files 모듈 가져옴, 파일을 Colab 환경에 업로드하거나 다운로드 할 때 사용
```
#### ◼️ file_uploaded = files.upload()
```
=> file.upload()는 사용자가 로컬 시스템에서 파일을 Google Colab 환경으로 업로드할 수 있도록하는 함수
```
## 🟣 One-Hot Encoding 기법으로 데이터 전처리
```
# 연속형(측정가능), 명목형(순서없는) 데이터로 분류
# 머신러닝 모델에 사용하기위하 범주형 데이터를 수치형 데이터로 변환하는 전처리 과정
# One-hot encoding을 통해 범주형데이터를 0, 1 로 변환
categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
data.head()
```
### ➕ 데이터 유형들
```
1. 연속형 데이터 Continuous Data
값들이 연속적으로 나타나는 데이터, 무한히 많은 값을 가지며, 측정 가능한 데이터
2. 범주형 데이터 Categorical Data
값들이 특정 범주나 그룹으로 구분된 데이터, 이산적인 값 가짐
수치적인 의미가 없고, 그저 분류의 목적으로 사용 (순서나 간격 X)
분석 시 one-hot encoding 등의 변환이 필요할 수 있음
3. 명목형 데이터 Nominal Data
명목형 데이터는 범주형 데이터의 한 유형으로, 순서가 없는 범주 데이터
각 값은 구분만 가능하고, 어떤 값이 다른 값보다 크다 작다 비교할 수 없다
4. 서열형 데이터 Ordinal Data
범주형 데이터의 한 유형으로, 값들 간의 순서는 존재하지만, 간격이 균등하지 않은 데이터
5. 수치형 데이터 Numerical Data
숫자로 표현되는 데이터로, 연속형 데이터와 이산형 데이터로 구분

```
#### ◼️ categorical_features = ['Channel', 'Region']
```
=> 범주형 데이터로 처리할 열 지정, 원핫 인코딩 통해 수치형 데이터로 변
```
#### ◼️ continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
```
=> 연속형 데이터로 처리할 열 지정
```
#### ◼️ for col in categorical_features:
```
=> for 사용해 리스트에 있는 각 열 순차 처리, 원핫 인코딩 적용하는 작업 반복
```
#### ◼️ dummies = pd.get_dummies(data[col], prefix=col)
```
=> pd.get_dummies는 범주형 데이터를 원핫 인코딩으로 변환하는 함수,
data[col]은 현재 처리 중인 범주형 열을 의미,
prefix=col은 더미 변수의 이름에 원래 열 이름을 접두사로 붙여, 새로운 열 이름 구분
```
#### ◼️ data = pd.concat([data, dummies], axis=1)
```
=> pd.concat() 함수는 데이터프레임 이어붙이는 함수, axis=1은 열방향으로 결
```
#### ◼️ data.drop(col, axis=1, inplace=True)
```
=> drop() 함수 특정 열을 데이터프레임에서 제거하는 함수,
col은 현재 처리 중인 범주형 열의 이름, 범주형 열을 삭제함
axis=1은 열을 기준으로 삭제한다는 의미, inplace=True는 원본 DF에서 직접 변경 수행
```
## 🟣 MinMaxScaler() 데이터 전처리
```
# 연속형 데이터의 전처리(스케일링:Scailing)
# 일정한 범위를 유지하도록 사이킷 런의 MinMaxScaler() 사용
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

data_transformed = pd.DataFrame(data_transformed, columns=data.columns)
data_transformed.head()
```
#### ◼️ mms = MinMaxScaler()
```
=> MinMaxScaler()는 사이킷런 sklearn.preprocessing에서 제공하는 데이터 전처리 클래스
데이터를 일정한 범위(0~1)로 변환 = '스케일링'
주로 데이터의 값 범위가 다를 때 학습 효율을 높이기 위해 사용
이 코드에서는 객체를 생성하는 코드
```
#### ◼️ mms.fit(data)
```
=> 스케일러에 fit() 메소드 적용해 데이터의 최대, 최소 계산
스케일링을 위해 데이터를 학습시킴, 데이터의 각 특징(컬럼)에 대한 최대 최소 기준으로 스케일링
fit() 메소드는 모델이나 변환기를 데이터에 맞게 학습시키는 기능
data는 DF나 배열 형태로 주어지며, mms는 이 데이터를 기준으로 최대 최소 학습
```
#### ◼️ data_transformed = mms.transform(data)
```
=> transform() 메소드 사용해 데이터를 스케일링
fit() 통해 학습한 값을 이용해 실제 데이터를 변환함, 각 컬럼을 0과 1 사이로 변환
transform()은 데이터를 변환하는 메소드로, fit()에서 학습된 정보를 바탕으로 변환 작업
```
#### ◼️ data_transformed = pd.DataFrame(data_transformed, columns=data.columns)
```
=> 변환된 데이터를 다시 DF로 변환
스케일링된 데이터를 새로운 판다스 DF로 저장, 기존 데이터의 컬럼 이름 유지
pd.DataFrame()은 판다스에서 데이터프레임을 생성하는 함
```
## 🟣 그래프 시각화(최적의 K값 찾는 엘보우 기법 Elbow Method)
```
# 데이터 전처리 완료후 .....
# K 값 추출 (타당성  평가)
Sum_of_squared_distances = []
K = range(1,10)                                     # K값을 1~10 까지 적용
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)                    # KMeans 모델 훈련
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()
```
#### ◼️ Sum_of_squared_distances = []
```
=> 빈 리스트 생성하여, 각 클러스터 개수에 대한 KMeans 모델의 SSE 값 저장
모델을 여러 번 실행하여, 각 클러스터 수에 따른 SSE 값을 차례로 저장할 공간을 만듦
```
#### ◼️ K = range(1,10)
```
=> 1부터 9까지의 값을 포함하는 range 객체 생성, 범위에 포함된 값 차례로 순회
```
#### ◼️ for k in K:
```
=> K는 순회할 수 있는 객체, k는 현재 순회 중인 값 저장하는 변
```
#### ◼️ km = KMeans(n_clusters=k)
```
=> KMeans 모델을 클러스터 개수 k로 설정하여 초기화함 
```
#### ◼️ km = km.fit(data_transformed)
```
=> 변환된 데이터를 사용해 KMeans 모델 훈
```
#### ◼️ Sum_of_squared_distances.append(km.inertia_)
```
=> 훈련된 KMeans 모델의 관성 inertia 값을 리스트에 추가함
.inertia_는 모델의 관성, 각 클러스터의 중심과 해당 클러스터 내의 점들 간의 거리 제곱 합
Sum of Squared Distances = 클러스터 내 데이터들이 얼마나 중심에 가까운지 평가하는 지표
```
#### ◼️ plt.plot(K, Sum_of_squared_distances, 'bx-')
```
=> 클러스터 개수 K와 클러스터 수에서의 SSE값을 그래프로 시각화
클러스터 개수와 SSE 값의 변화를 통해 엘보우 포인트를 찾음
엘보우 포인트 = 그래프에서 SSE 감소가 급격히 줄어드는 지점 = 최적의 클러스터 개수
```
## 🟣 그래프 시각화 2 
#### ◼️ from yellowbrick.cluster import KElbowVisualizer
```
=> yellowbrick 라이브러리에서 KElbowVisualizer 모듈 가져옴
엘보우 기법을 시각화하는데 필요한 클래스 가져옴 
```
#### ◼️ iris = datasets.load_iris()
```
=> iris 데이터셋 불러옴, 사이킷런에서 제공하는 데이터를 불러오는 함수
4개의 특징: 꽃잎, 꽃받침의 길이, 너비 포함 4개의 특징 + 3개의 품종(클래스)
```
#### ◼️ X = iris.data
#### ⬛ y = iris.target
```
=> 특징 데이터 X와 타겟 데이터 y 분리
X는 데이터셋의 특징 데이터를 담고, y는 데이터에 해당하는 클래스(품종) 나타냄
X만을 클러스터링에 사
```
#### ◼️ df = pd.DataFrame(X, columns = iris.feature_names)
```
=> X 데이터를 판다스 DF으로 변환, 각 컬럼에 이름을 지정 
```
#### ◼️ k=0
```
=> 변수 k에 값을 0으로 초기화
```
#### ◼️ kmeans = KMeans(n_clusters=k, random_state=7)
```
=> 클러스터링 모델을 초기화
n_clusters=k는 클러스터 수 지정,
엘보우 기법에서는 k가 아닌 시각화 도구에서 자동으로 최적의 k 값 찾음
random_state는 랜덤하게 선택된 클러스터 중심이 매번 동일하게 선택되도록 고정 역할
```
#### ◼️ visualizer = KElbowVisualizer(kmeans, k=(1,10), timings=False)
```
=> 엘보우 기법을 시각화하는 도구로, 다양한 클러스터 개수에 대해 KMeans 모델 훈련,
클러스터 개수에 따른 성능 변화를 시각화, k=(1,10)은 1부터 10까지 시도하겠다는 것
timings=False는 각 모델 훈련에 걸린 시간을 시각화하지 않겠다는 설정
```
#### ◼️ visualizer.fit(df)
```
=> df 데이터를 KMeans 모델에 훈련시키면서 다양한 클러스터 수에 대해 성능 평가
fit()은 KMeans 모델을 훈련시키는 메소드 
```
## 🟣 PCA: 주성분 분석 (Principal Component Analysis)
### 고차원 데이터를 더 낮은 차원으로 변환하여 데이터를 압축하는 기법
```
#### PCA : 주성분 분석(Principal Component Analysis)
#################################################################
#### 특성 p개를 2~#개 정도로 압축해서 데이터를 시각화하여 봄
#### 유사한 특성(Feature)을 처리: 고차원 -> 저차원
```
```
주로 데이터의 시각화 및 차원 축소에 사용
PCA는 데이터의 분산을 최대한 유지하면서, 유사한 특성들 간의 상관성을 줄이고,
정보를 잃지 않으면서 차원을 줄이는 것을 목표로 함

이 기법은 데이터를 새로운 좌표계로 반환하는데
이 좌표계의 축들을 데이터의 분산이 가장 큰 방향을 따르게 됨
고차원 데이터를 저차원으로 변환하더라도 중요한 정보는 최대한 유지됨
```
### ➕ PCA의 주요 개념과 과정
#### ◼️ 1. 고차원 데이터란
```
고차원 데이터: 데이터의 많은 특성을 가진 데이터
차원이 너무 크면 계산이 복잡해지고, 시각화나 해석이 어려워짐
고차원 데이터는 중복된 데이터 존재하고
몇몇 특성은 서로 강한 상관관계 가지고, 실제로는 중복된 정보를 제공하는 경우가 많음
```
#### ◼️ 2. 차원 축소의 필요성
```
차원 축소는 데이터를 분석하고 시각호하는데 중요한 도구
- 모델의 학습 속도를 높이고, 과적합 overfitting을 방지하는데 도움
- 데이터의 시각화가 가능, 그래프로 쉽게 표현 가능함
- 데이터의 노이즈를 줄여 더 중요한 특징만을 남기게 됨
```
#### ◼️ 3. 주성분이란
```
주성분 Principal Components은 원본 데이터에서 가장 중요한 정보를 설명하는 새로운 축들
이 축들은 데이터의 분산이 최대한 많이 설명되는 방향을 가리킴
```
#### ◼️ PCA의 단계
##### 1️⃣ 데이터 정규화
```
각 특성의 스케일이 다를 수 있으므로, 정규화나 표준화 진행
특성들의 중요도를 동일하게 하기 위함
```
##### 2️⃣ 공분산 행렬 계산
```
각 특성 간의 공분산 계산하여, 특성들이 어떻게 변하는지 상관관계 파악
```
##### 3️⃣ 고유값 및 고유벡터 계산
```
공분산 행렬에서 고유값과 고유백터 계산
고유값이 클수록해당 고유벡터가 데이터의 분산을 많이 설명 
```
##### 4️⃣ 주성분 선택
```
고유값이 큰 고유벡터를 선택하여 주성분으로 사용, 이를 기준으로 새로운 차원 변환
```
##### 5️⃣ 차원 축소
```
원하는 수의 주성분을 선택해 데이터를 저차원으로 변환, 중요한 정보 유지
```
## 🟣 라이브러리 
```
# library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.cluster import DBSCAN                  #밀도 기반 군집 분석
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize         # Changed the import to be from sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
```
#### ◼️ from sklearn.cluster import DBSCAN
```
DBSCAN은 밀도 기반 클러스터링 알고리즘
밀도가 높은 영역을 클러스터로 정의하고, 낮은 영역에는 노이즈나 외부 포인트로 간주
```
#### ◼️ from sklearn.preprocessing import StandardScaler
```
StandardScaler는 데이터를 평균 0, 분산 1로 표준화하는데 사용
모든 특성이 동일한 스케일에서 작동하도록 함 = 데이터 전처리 단계
```
#### ◼️ from sklearn.preprocessing import normalize
```
normalize는 데이터를 정규화하는 함수
벡터의 크기를 1로 만들어, 각 특성 간의 스케일 차이를 줄임 
```
#### ◼️ from sklearn.decomposition import PCA
```
차원 축소 기법 PCA, 데이터의 분산을 최대한 설명할 수 있는 방향으로 차원 축소
```
#### ◼️ from sklearn.preprocessing import MinMaxScaler
```
MinMaxScaler는 데이터를 0과 1사이 값으로 변환하는데 사용
각 데이터가 일정한 범위 내에서 속하게 함
```
#### ◼️ from sklearn.cluster import KMeans
```
KMeans는 군집(클러스터링) 알고리즘
데이터를 k개의 클러스터로 나누는 알고리즘,
각 클러스터의 중심을 계산하여 데이터 포인트들을 가장 가까운 중심으로 할당 
```
#### ◼️ from sklearn imort svm
```
SVM은 분류와 회귀에 모두 사용되는 지도 학습 알고리즘
데이터를 분리하는 최적의 초평면을 찾는 것이 목표
주로 분류 문제에 사용되며, 선형 또는 비선형적 데이터도 처리할 수 있음
```
#### ◼️ from sklearn import metrics
```
metrics 모듈은 모델의 성능을 평가하는데 사용
```
#### ◼️ from sklearn import datasets
```
datasets 모듈은 머신러닝에서 자주 사용되는 샘플 데이터셋을 제공
```
#### ◼️ from sklearn.model_selection import train_test_split
```
train_test_split은 데이터를 학습 데이터와 테스트 데이터로 나누는 함수
데이터의 일부를 테스트 데이터로 남겨둠으로써,
모델이 학습되지 않은 데이터를 얼마나 잘 예측하는지 평가 
```
#### ◼️ from sklearn.svm import SVC
```
SVC Support Vector Classifier 은 SVM 분류 알고리즘
데이터를 여러 클래스 중 하나로 분류하는데 사용
다양한 커널을 선택할 수 있으며, 비선형 데이터도 처리할 수 있음 
```
## 🟣 데이터 로드
```
# data
from google.colab import files
file_uploaded=files.upload()

#X= pd.read_cas('../chap3/data/credit card.csv')
X = pd.read_csv('credit card.csv')
X = X.drop('CUST_ID', axis=1)           # 불러온 데이터에서 'CUST_ID'열(colum) 삭제
X.fillna(method='ffill', inplace=True)  # 결측값을 앞의 값으로 채움
print(X.head())
```
#### ◼️ X = X.drop('CUST_ID', axis=1)
```
=> X DF에서 열을 기준으로 CUST_ID 열 삭제 
```
#### ◼️ X.fillna(method='ffill', inplace=True)
```
=> 데이터 프레임 내의 결측값을 앞의 값으로 채움
method="ffill"은 foward fill로, 결측값을 바로 앞에 있는 값으로 채움
fillna()는 결측값을 채우는 판다스 함수, inplace=True는 원본 DF에 바로 적용
```
## 🟣 전처리 & 데이터를 2차원으로 축소
```
# 전처리 & 데이터를 2차원으로 축소
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)            # 평균이 0, 표준편차가 1이되도록 데이터 크기를 조정

X_normalized = normalize(X_scaled)            # 데이터가 가우스 분포를 따르도록 정규화
X_normalized = pd.DataFrame(X_normalized)     # 넘파이 배열을 -> 데이터프레임으로 변환

pca = PCA(n_components = 2)                   # 2차원으로 축소 선언
X_principal = pca.fit_transform(X_normalized) # 차원 축소 적용
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head()) 
```
#### ◼️ scaler = StandardScaler()
```
StandardScaler() 객체 생성, 데이터 표준화하는 객체
각 특성의 평균을 1, 표준편차 1로 맞춰서 스케일링 조정
```
#### ◼️ X_scaled = scaler.fit_transform(X)
```
X 데이터에 대해 StandardScaler 적용해 스케일링
fit_transform() 데이터를 표준화하고, 그 결과를 반환
fit()은 데이터를 학습, transform() 데이터를 학습한 기준에 따라 반환
fit_transform()은 데이터를 학습과 변환을 한 번에 수행하는 메소드 
```
#### ◼️ X_normalized = normalize(X_scaled)
```
스케일링된 데이터를 정규화
normalize() 벡터의 크기를 1로 정규화,
이 과정은 데이터의 분포를 가우스 분포에 가깝게 만들거나, 각 특성의 스케일을 맞추는데 사용
데이터의 각 행을 벡터의 크기가 1로 되도록 정규화 
```
#### ◼️ X_normalized = pd.DataFrame(X_normalized)
```
정규화된 넘파이 배열을 데이터 프레임으로 변환, 데이터 처리 더 쉽게 수행
```
#### ◼️ pca = PCA(n_components=2)
```
주성분 분석 객체를 생성, 차원을 2차원으로 축소
PCA 알고리즘을 통해 고차원의 데이터를 두 개의 주성분으로 축소하기 위한 객체 생성
```
#### ◼️ X_principal = pca.fit_transform(X_normalized)
```
정규화된 데이터에 PCA 적용해 2차원으로 축소, 넘파이 배열로 결과를 반환
반환된 결과는 차원이 축소된 데이
```
#### ◼️ X_principal.columns = ['P1', 'P2']
```
데이터 프레임의 열 이름을 P1, P2로 설정 
```
## 🟣 DBSCAN 모델 생성 및 결과의 시각화
```
db_default = DBSCAN(eps = 0.0375, min_samples = 3).fit(X_principal)   # 모델 생성 및 훈련
labels = db_default.labels_
# 각 데이터 포인트에 할당된 모든 클러스터 레이블의 넘파일 배열을 Labels에 저장
```
#### ◼️ db_default = DBSCAN(eps=0.0375, min_samples=3).fit(X_principal)
```
DBSCAN 모델 생성, X_principal 데이터에 대해 훈련
eps는 데이터 포인트 간의 최대 거리 지정하여, 이 거리 내의 데이터 포인트들이 밀집 되어있으면 클러스터
min_samples=3은 클러스터를 형성하기 위해 한 데이터 포인트 주변에 최소 3개의 데이터 포인트가 있어야 함 
```
#### ◼️ labels = db_default.labels_
```
모델에서 각 데이터 포인트에 할당된 클러스터 레이블을 저장, 넘파이 배열로 반환
labels 속성은 DBSCAN이 각 데이터 포인트에 할당한 클러스터 레이블을 나타냄
해당 클러스터 번호로, 그리고 노이즈로 간주된 포인트는 -1로 표시
DBSCAN 모델 객체의 속성으로, 넘파이 배열로 나타남
```
#### ➕ DBSCAN의 특징
```
1. 밀도 기반 클러스터링
밀도가 높은 영역을 클러스터로 인식, 밀도 낮은 영역의 포인트는 노이즈로 간주
2. 노이즈 포인트
밀도가 낮은 영역에 속한 포인트는 클러스터에 속하지 않고, 노이즈 포인트로 분류
DBSCAN 레이블에서 -1로 표시되는 데이터가 노이즈 포인트
3. 비구형 클러스터링
KMeans 와 달리, DBSCAN은 비선형 데이터 구조에서도 클러스터링이 잘 작
```
## 🟣 그래프 시각화 1 
```
colours = {}        #출력 그래프의 색상을 위한 레이블 생성
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'

cvec = [colours[label] for label in labels]   # 각 데이터 파인트에 대한 색상 벡터 생성

r = plt.scatter(X_principal['P1'], X_principal['P2'], color ='y');
g = plt.scatter(X_principal['P1'], X_principal['P2'], color ='g');
b = plt.scatter(X_principal['P1'], X_principal['P2'], color ='b');
k = plt.scatter(X_principal['P1'], X_principal['P2'], color ='k');   # plot의 범례(legend) 구성

plt.figure(figsize =(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)
# 정의된 색상 벡터에 따라 X축에 P1, Y축에 P2 플로팅(plotting)

plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))  # legend구축
plt.show()
```
#### ◼️ 출력 그래프의 색상을 위한 레이블 생성
```
colours = {}  
colours[0] = 'y'
colours[1] = 'g'
colours[2] = 'b'
colours[-1] = 'k'
```
```
각 클러스터 레이블에 해당하는 색상 설정, DBSCAN 모델에서는 클러스터 번호로 0,1,2를 지정하고
-1은 노이즈 데이터 의미, colors 딕셔너리 자료형으로, 레이블을 키, 색상 값을 값으로 사용
```
#### ◼️ cvec = [colours[label] for label in labels]  
```
각 데이터 포인트의 클러스터 레이블에 따라 colors 딕셔너리에서 적합한 색상 가져와 벡터 cvec 생성
리스트 내포 사용해 각 데이터 포인트의 클러스터 레이블에 맞는 색상 찾아 배열로 저장 
```
#### ◼️ plot의 범례(legend) 구성
```
r = plt.scatter(X_principal['P1'], X_principal['P2'], color='y')
g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g')
b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b')
k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k') 
```
```
범례를 구성하기 위해 임의의 산점도 그램, x축에 P1, y축에 P2 값 사용
plt.scatter()는 X,Y좌표 기반으로 산점도 그리는 함수
```
#### ◼️ plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec) 
```
새로운 플롯 생성하고, 데이터 포인트 X_principal의 P1값을 X축, P2 값을 Y축으로 산점도 그
```
#### ◼️ plt.legend((r, g, b, k), ('Label 0', 'Label 1', 'Label 2', 'Label -1'))  
```
각 레이블이 어떤 색상에 해당하는지 명시적으로 보여주는 역할
```
#### 🟥 첫번째 그래프 해석
```
모든 데이터 포인트가 검은색으로 표시되어 있어, 데이터 간의 구분이 전혀 이루어지지 않음
color='k'로 모든 데이터 포인트가 노이즈 색상인 검은색으로 표시 
```
#### 🟥 두번째 그래프 해석
```
노란색(0): 데이터의 대부분이 하나의 큰 클러스터로 분류
녹색(1) 과 파란색(2): 몇몇 데이터 포인트들이 작은 클러스터
검은색(-1): 노이즈 포인
```
## 🟣 그래프 시각화 2
```
db = DBSCAN(eps = 0.0375, min_samples = 50).fit(X_principal)
labels1 = db.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels1]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ]

r = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[0])
g = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[1])
b = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[2])
c = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[3])
y = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[4])
m = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[5])
k = plt.scatter(
    X_principal['P1'], X_principal['P2'], marker ='o', color = colors1[6])

plt.figure(figsize =(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c = cvec)
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints = 1,
           loc ='upper left',
           ncol = 3,
           fontsize = 8)
```
#### ◼️ db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
#### ◼️ labels1 = db.labels_
```
DBSCAN 클러스터링 모델을 생성하여 X_principal 데이터를 클러스터링
eps는 포인트 간의 최대 거리, min_sample은 최소 데이터 수
각 데이터가 어느 클러스터에 속하는지 노이즈인지 labels1에 저장
DBSCAN()은 DBSCAN 클러스터링 수행하는 객체, fit() 은 데이터를 클러스터링,
labels_는 클러스터 레이블 반환
``` 
#### ◼️  cvec = [colours1[label] for label in labels1]
```
클러스터 레이블에 따라 색상을 택해 cvec에 저장
각 데이터 포인트의 레이블에 해당하는 색상 colours1에 가져와 리스트로 만듦
```
#### ◼️ 범례 지정
```
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=8)
```
```
각 클러스터 레이블에 대응하는 색상 및 레이블을 지정하고, 범례의 위치를 왼쪽 위로 설정하며,
한줄에 3개의 컬럼 ncol=3 사용하여 범례를 정렬 (열 수 설정)
```
#### 🟥 그래프 해석 1
```
파라미터 eps와 min_samples 설정으로 인해 대부분 데이터 포인트가 노이즈로 분류되었기 때문임
```
#### 🟥 그래프 해석 2
```
클러스터: 다양한 색상으로 소수의 데이터 포인트들이 클러스터로 그룹화
노이즈: 검은색으로 표시 = 노이즈
```
#### 🟦 이전 그래프와의 차이
```
min_samples 값의 차이: 첫 번째는 3, 두 번째는 50
매우 제한된 영역에서만 클러스터가 형성, 대부분 데이터는 크러스터의 밀도 조건을 만족 X -> 노이즈
```
## ➕ 선형 VS 비선형 데이터의 차이
### ◼️ 선형 데이터 Linear Data
```
데이터 포인트가 직선적 관계를 따르는 데이터
두 변수 간의 관계가 직선으로 표현될 수 있다면 선형 데이터
- 데이터가 일정한 비율로 변하면서, 직선 or 평면으로 설명 가능
- 선형 회귀나 PCA와 같은 선형 변환 기법으로 데이터 분석이 가능
- 데이터의 변화가 단순하고 예측 가능성이 높음
ex) 키와 몸무게, 수학적 함수 관계
```
### ◼️ 비선형 데이터 Non-linear Data
```
데이터 간의 관계가 비직선적으로, 복잡한 곡선이나 패턴을 따르는 데이터
두 변수 간의 관계가 단순히 직선이나 평면으로 설명될 수 없는 경우
- 데이터가 복잡하고 다양한 방식으로 변할 수 있으며, 비직선적 패턴을 나타냄
- 데이터 간의 관계가 곡선 또는 더 복잡한 구조로 나타나, 비선형 변환 기법 사용해야 함
- 예측이 어렵고, 데이터의 구조가 복잡해 고차원 공간에서 잘 설명됨
```
## ➕ PCA vs t-SNE 
### ◼️ PCA
```
선형 데이터를 잘 처리함, 선형 변환을 통해 고차원 데이터를 저차원으로 투평할 때
데이터의 분산이 가장 큰 방향으로 주성분을 선택하여 차원을 축소
비선형적 관계까 있는 데이터에서는 PCA가 잘 작동하지 않을 수 있음
```
### ◼️ t-SNE
```
비선형 데이터를 다루는데 적합한 기법, 데이터가 고차원에서 복잡한 구조를 가지는 경우,
국소적인 구조를 보존하면서 저차원으로 축소하여 시각적으로 파악하기 쉽게 만ㄷ름
비슷한 데이터 포인트를 가까운 위치에 배치하므로, 비선형적 구조가 반영된 시각화가 가능해짐
ex) 숫자 데이터셋 Digits의 경우 픽셀 값의 분포가 비선형적 구조, 숫자 이미지 간의 관계는 비선형적 
```
|특징|선형 데이터|비선형 데이터|
|--|--|--|
|관계|직선적 (변수 간 선형적 관계)|복잡한 곡선 or 비직선적 관계|
|차원축소기법|PCA (선형적 투영)|t-SNE (비선형적 구조 보존)|
|예시|키와 몸무게 관계|자연현상에서 나타나는 복잡한 데이터|
|장점|간단하고 예측이 쉬움|복잡한 데이터 구조 설명 가능|
|단점|복잡한 데이터 설명 못함|계산 비용이 들고, 전역적인 구조 설명 부|

