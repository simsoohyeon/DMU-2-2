![image](https://github.com/user-attachments/assets/30babadf-0345-4855-b5f2-ba6afa3f3aaa)<div align = "center">
  <h2> SVM (Support Vector Machine).ipynb </h2>
</div>
  
```
# library
import numpy as np
=> 파이썬에서 수치 계산을 위한 강력한 배열 연산 라이브러리
import pandas as pd
=> 테이블 형식 데이터를 다루고 분석하기 위한 판다스 라이브러리, 주로 DF 처리
import matplotlib.pyplot as plt
=> 데이터 시각화를 위해 사용되는 pyplot 모듈 가져옴, 주로 그래프나 차트를 그리는데 사용
import tensorflow as tf
=> 머신러닝 모델을 구축하고 학습시키는데 사용되는 Tensorflow 라이브러리 가져옴

from sklearn import svm
=> SVM 알고리즘 제공하는 scikit-learn의 svm 모듈 가져옴, 주로 분류와 회귀 작업에 사용 
from sklearn import metrics
=> 머신러닝 모델의 성능을 평가하는 함수들이 포함된 metrics 모듈 가져옴
from sklearn import datasets
=> 샘플 데이터셋 제공하는 datasets 모듈 가져옴
from sklearn.model_selection import train_test_split
 => 데이터를 학습용과 테스트용으로 나누기 위한 train_test_split 함수 가져옴
from sklearn.svm import SVC
=> svm 사용한 분류 작업을 위한 SVC 클래스 가져옴
import os
=> 운영체제와 상호작용하기 위한 os 모듈 가져옴, 파일처리나 환경변수 설정
# 환경변수를 사용하여 로깅을 제어
# 기본 0, INFO 로그로 필터링 1, WARNING 로그로 필터 2, 로그를 추가로 필터링 3 설정
os.environ['TF_CPP_WIN_LOG_LEVEL'] = '3'
=> 환경 변수를 3으로 설정, TensorFlow의 로그출력을 제어하는 역할
- 0: 디버그 정보 포함한 모든 로그가 출력
- 1: 정보, 경고, 오류만 출력
- 2: 경고와 오류만 출력
- 3: 오류만 출력, 가장 제한적
```
```
# <<시험문제>> sklearn에서 제공하는 iris 데이터 호출
iris = datasets.load_iris()
=> datasets 모듈에서 제공하는 load_iris() 함수는 붗꽃 데이터셋을 호출하는 함수
4개의 특징을 가진 150개의 붗꽃 데이터와 2개의 클래스(레이블)로 구성됨
iris는 데이터와 목표 값을 모두 포함한 객체로, iris.data는 특징 X,
iris.target은 분류 레이블, 종류 y를 나타냄

# sklearn에서 제공하는 model_selection 패키지에서 제공하는 train_test_split() 메소드를 활용
# KNN에서는 iris.data = X, iris.target = y 역할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=42)
=> train_test_split 함수는 데이터를 학습용 train과 테스트용 test으로 분리하는데 사용
- iris.data: 입력 데이터, 특징 값, X역할
- iris.target: 정답 레이블, 붗꽃의 종류, y역할
- test_size=0.3 전체 데이터셋의 30%를 테스트 데이터로 할당, 나머지는 70%는 학습용으로 사용
- random_state=42 랜덤으로 데티러르 나누지만, random_state 설정하면 데이터 분할이
매번 동일하게 이루어짐(재현 가능성을 위해)
#X_train.shape
=> 학습용 입력 데이터의 형상을 확인하는 코드
shape는 데이터의 차원(행과 열) 반환하며 (105,4)
#X_test.shape
=> 테스트용 입력 데이터의 형상을 확인하는 코드
테스트 데이터로 사용되는 행과 특증의 수 반환 (45,4)
#y_train.shape
=> 학습용 목표 레이블(붗꽃 종류)의 형상 확인코드
y_train의 행 수는 X_train의 행 수와 일치해야 함 (105,)
y_test.shape
=> 테스트용 목표 레이블의 형상 확인 코드
y_test의 행 수는 X_test의 행 수와 일치함 (45,)
```
### 🟣 관련 문법
```
- shape: 배열이나 DF에서 데이터의 차원을 나타내는 속성, 배열의 각 차원의 크기 튜플로 반환
- train_test_split(): 데이터를 랜덤하게 학습과 테스트 세트로 나누는 유틸리티 함수
test_size와 random_state과 같은 매개변수를 통해 분할 비율과 랜덤성 제어 
```

```
# SVM model accuracy
# svm = svm.SVC(kernel='linear', c=1.0, gamma = 0.5) 참고_감마는 리니어에 불필요, RBF에 필
# svm.fit(X_train, y_train)

model = svm.SVC(kernel='linear', C=10.0)
=> SVC Support Vector Classifier 객체 생성
- kernel='linear' 선형 커널을 사용하여 데이터를 분류, 데이터가 선형적으로 구분되는 경우
- C=10.0: 오류에 대한 패널티 조절하는 하이퍼파라미터
C값이 클수록 모델은 오류를 허용하지 않으며(과적합 가능성), 값이 작으면 오류를 더 허용해 일반화
# kenerl = "linear" : 선형 커널을 사용해 데이터를 분류, 3D="RBF"
=> rbf는 비선형 데이터를 처리하는데 사용되는 방사기저함수를 뜻함, 3차원 이상의 데이터
# c=1.0 : 오류에 대한 패널티를 조절하는 하이퍼파라미터(값이 크면 ..?)
=> C값이 크면 모델은 오류를 줄이려고 노력하므로 데이터에 더 민감하게 반응, 과적합 위험 증가
반대로 C값이 작으면 오류를 더 허용하여 모델이 덜 복잡해지며 과적합을 줄이고 일반화 성능 높임
# gamma=0.5 : RBF 커널 (Radial Basis Function) 방사 기저 함수: 비선형
= gamma는 RBF 커널에서 사용되는 하이퍼파리머터로, 데이터 포인트트에 영향을 결정
gamma가 클수록 개별 포인트의 영향이 커지고, 작을수록 넓은 영역이 영향을 미침
<<⚠️ 선형 커널에서는 gamma 매개변수가 필요하지 않>> 
model.fit(X_train, y_train) # 훈련 데이터를 사용해 SVM 분류리를 훈련
=> 훈련 데이터를 사용해 SVM 모델 학습 fit, 이 과정에서 모델은 데이터 보고 분류 경계 찾음

predictions = model.predict(X_test) # 훈련된 모델을 사용하여 테스트 데이터에서 예측
=> 학습된 SVM 모델을 사용해 테스트 데이터 X_test에 대한 예측 수행
predict()는 입력된 데이터의 각 샘플에 대해 모델이 예측한 클래스 레이블 반환
score = metrics.accuracy_score(y_test, predictions)
=> 정확도 계산 코드, accuracy_score()는 테스트 데이터의 실제 레이블과 모델의 예측값 비교
print('정확도: {0:0.3f}'.format(score)) # 0.3f 소수셋째자리까지 vs 0.2f 소수둘째자리까지
=> 정확도 score를 소수점 셋째 자리까지 포맷하여 출력
```
### 🟣 관련 문법
```
- kernel="linear": SVM에서 사용하는 커널 함수, 데이터의 분포가 선형일 때 선형 커널 사용,
비선형일 때는 RBF 같은 비선형 커널 사용
- C: SVM 하이퍼파라미터, 데이터 분류시 오류를 허용할지 여부 결정
- fit(): 주어진 데이터로 모델을 학습시키는 메소드
- predict(): 학습된 모델을 사용해 새로운 데이터의 결과를 예측
- accuracy_socre(): 모델의 예측 성능을 평가하기 위해 사용되는 함수, 정확도 계산
- format(): 문자열 내에서 변수를 특정 형식으로 포맷하기 위해 사용 
```
### 🔴 감마가 클수록 개별 포인트에 영향, 작을수록 넓은 영역에 영향을 미치는 이유
```
Gamma는 비선형 데이터를 다룰 때 사용되는 중요한 하이퍼파라미터
RBF커널은 데이터 포인트 주위에 방사형으로 영향을 미치는 함수로, 데이터의 분포에 다라
분류 경계를 비선형적으로 만듦, 데이터 포인트가 미치는 영향 범위 조정 
```
#### ◼️ 감마가 클 때
```
감마 값이 클수록 각 데이터 포인트의 영향 범위가 짧고, 국소적으로만 영향을 미침
한 포인트의 영향이 근처의 포인트에만 미치고, 더 멀리 있는 포인트들에게는 거의 영향주지 않음
감마가 크면 모델이 세부사항에 집중하게 되어, 학습 데이터에 지나치게 맞추려 하므로 과적합 위험
```
#### ◼️ 감마가 작을 때 
```
감마 값이 작으면 각 포인트의 영향 범위 넓어짐
하나의 데이터 포인트가 더 넓은 영역에 걸쳐 영향을 미치게 되고, 여러 포인트의 영향을 한꺼번에 고려
모델이 전체 데이터의 일반적인 분포를 더 잘 학습하게 되어, 과적합 위험을 줄이고 일반화 성능 높임
=> 감마 값이 작을수록 분류 경계가 더 부드럽게 형성, 감마 값이 클수록 매우 구체적이고 복잡
```
### 🔴 선형 커널에 Gamma가 필요하지 않은 이유
```
데이터가 직선이나 평면으로 구분할 수 있을 때 적합
데이터를 고차원 공간으로 변환할 필요가 없으며, 주어진 특징 공간에서 직선적인 경계를 그려 데이터 분류
=> 단순히 직선 경계로 분류하기 때문에 감마 값을 필요로 하지 않음 
```
```
print(y_test)
=> 테스트 데이터셋의 실제 레이블을 출력
print(predictions)
=> SVM 모델이 예측한 테스트 데이터에 대한 결과 출력, model.predict(X_test)를 통해 얻어진 모델의 예측

print(metrics.classification_report(y_test, predictions))
# accuracy: 정확도 TP+TN
# precision: 정밀도 (예측한 것을 실제로 맞춘 비율) 
# recall: 재현율 (실제를 예측에서 맞춘 비율)
=> classification_report()는 분류문제에서 정확도, 정밀도, 재현율, F1-Score를 포함해 성능 지표 출력
```

<div align="center">
  <h2> SVM2 </h2>
</div>

```
# iris = datasets.load_iris()
=> 라이브러리에서 제공하는 iris 데이터셋 불러옴

iris.feature_names
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
=> 데이터셋의 특성 이름, 컬럼 이름 가져옴 

df = pd.DataFrame(iris.data, columns=iris.feature_names) # 데이터 프레임을 만들어줌
=> .DataFrame() 함수 사용해 데이터프레임 생성
iris.data 사용해 데이터를 넣고, columns파라미터에 iris.feature_names 넣어 컬럼 이름 설정

df['target'] = iris.target # 'target' feature을 만들어줌
=> 데이터프레임 df에 target이라는 새로운 컬럼 추가, 꽃의 종류 나타냄

iris.target_names # 'setosa'=0, 'versicolor'=1, 'virginica'=2
=> 꽃의 종 이름 가져옴, 이 값들은 숫자로 표시
np.array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
=> 배열을 사용해 꽃의 이름 저장, <10은 문자열의 최대 길이가 10인 유니코드 데이터형 의미
이 부분은 iris.target_names가 이미 있으므로 추가적으로 표시 X 

# iris의 target_names을 apply하여 이름을 나타내어줌
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
=> apply() 함수와 lambda함수 사용하여 target 컬럼의 숫자값을 꽃의 종 이름으로 변환환

df0 = df[df.target==0] => target 값이 1인 데이터 추출해 df0에 저장
df1 = df[df.target==1]
df2 = df[df.target==2]

plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
=> df0에서 두 값으로 산점도 그림
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='+')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)') 
```
![image](https://github.com/user-attachments/assets/e13bb2dc-d83a-462b-8ac9-cb51b239cc4f)
```
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='+')
=> 두 종의 꽃잎 특성이 다름을 시각적으로 보여주며, 두 종이 잘 분리됨 
```
![image](https://github.com/user-attachments/assets/8ffacd35-9355-439a-aaed-cee7ab05cb21)

```
X = df.drop(['target', 'flower_name'], axis='columns')
=> DF에서 두 열을 제거하고, 나머지 데이터를 X에 저장, X는 특성=입력 데이터
y = df.target
=> df의 target, 꽃의 종을 y에 저장, 레이블 데이터, 출력 데이터

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
=> 전체 데이터셋의 20% 테스트용으로 할당, 나머지는 80%는 학습용으로 용도

model = SVC(C=10)
=> SVC 모델 생성, C는 SVM의 하이퍼파라미터로, 오류를 허용하는 정도
C=10은 오류에 대해 엄격하게 처리하는 하드 마진을 설정, 데이터에 과적합될 가능성 커짐
C값이 클수록 하드마진=오류 허용 안 함, 작을수록 소프트마진=오류 허용
# C의 값이 클수록 하드마진(오류 허용 안 함), 작을 수록 소프트마진(오류를 허용함)
# shift tab으로 SVC 확인해보기(여러 파라미터들이 있음)

model.fit(X_train, y_train)
=> SVC 모델을 학습 데이터를 사용하여 학습 

model.score(X_test, y_test)
=> 전체 예측에서 맞춘 비율 반환 
```

<div align="center">
  <h2>SVM 3</h2>
</div>

```
model = svm.SVC(kernel='rbf', gamma='scale') # kernel="linear" : 선형 커널을 사용해 데이터를 분류, 3D="RBF"
=> 비선형적인 데이터를 처리하는 SVC 모델 생성
gamma="scale"은 각 특징의 분산에 따라 적절한 gamma 값을 자동으로 설정
이 값이 크면 데이터 포인트의 영향 범위가 좁아지고, 값이 작으면 넓은 영역에 영향 미침

model.fit(X_train, y_train) # 훈련 데이터를 사용하여 svm 분류기를 훈련
```
```
# 테스트 데이터로 예측
y_pred = model.predict(X_test)
=> SVM모델을 사용해 테스트 데이터에 대한 예측 수행
predict()함수는 입력 데이터를 기반으로 모델이 예측한 레이블 반환
여기서 y_pred는 모델이 예측한 붗꽃의 종, X_test는 테스트 데이터의 특징값 -> 샘플 종 예측 

print(y_pred) 
=> 모델이 예측한 값 출력, 테스트 데이터에 대한 예측 결과 포함
실제 레이블 y_test와 비교해 모델의 성능 평가 

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification_report: \n", metrics.classification_report(y_test, y_pred))
=> classification_report(y_test, y_pred)는 다양한 성능 지표를 포함하는 리포트 반
```
### 🔴 학습 데이터와 Decision boundary Visualization(결정 경계 시각화)
```
def plot_decision_boundary(X,y,model):
=> 결정 경계 시각화 함수 정의, 입력 데이터, 레이블, 학습된 모델 받음
  h = 0.02 # 결정경계의 해상도, 값이 작을수록 더 정밀한 경계가 그려짐 
  x_min, x_max = X[:,0].min() -1, X[:, 0].max() +1
  y_min, y_max = X[:,1].min() -1, X[:, 1].max() +1
  => 데이터의 첫 번째, 두 번째 특성 받아 최소 최대 값을 구한 뒤,
  각각 -1과 +1 추가해 경계 바깥에 충분한 여백 설정 
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  => meshgrid는 2차원 평면 상에서 그리드를 생성하는 함수, xx, yy는 그리드의 좌표값 배열
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  => 그리드 상의 모든 점들에 대해 학습된 SVM 모델로 예측을 수행
  두 배열을 1차원으로  펼친 후, 이들을 결합해 모든 점에 대해 예측 수행하도록 배열 만듦
  Z = Z.reshape(xx.shape)
  => 예측된 결과 Z를 그리드 형태로 다시 재배열, 경계를 그릴 수 있는 2차원 배열 완성
  plt.contourf(xx, yy, Z, alpha=0.8) # 투명도 조절 
  => 등고선 플롯 사용해 결정 경계 시각화, 각 클래스가 나뉘는 영역 구분해 보여줌
  plt.scatter(X[:,0],X[:,1],c=y,edgecolors="k", marker='o') # 데이터 포인트 테두리 k
  => X[:,0]은 X축에 해당하는 꽃받침 길이, X[:,1]은 Y축에 해당하는 꽃받침 너비
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.title('SVM with RBF Kernel')
  plt.show()

# 결정경계 시각화
plot_decision_boundary(X_train, y_train, model) 
```
![image](https://github.com/user-attachments/assets/c89a1d77-5744-4da7-90b5-9aea0157bc90)




