<div align = "center">
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
iris 

# sklearn에서 제공하는 model_selection 패키지에서 제공하는 train_test_split() 메소드를 활용
# KNN에서는 iris.data = X, iris.target = y 역할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.3, random_state=42)

#X_train.shape
#X_test.shape
#y_train.shape
y_test.shape
```


```
# SVM model accuracy
# svm = svm.SVC(kernel='linear', c=1.0, gamma = 0.5) 참고_감마는 리니어에 불필요, RBF에 필
# svm.fit(X_train, y_train)

model = svm.SVC(kernel='linear', C=10.0)
# kenerl = "linear" : 선형 커널을 사용해 데이터를 분류, 3D="RBF"
# c=1.0 : 오류에 대한 패널티를 조절하는 하이퍼파라미터(값이 크면 ..?)
# gamma=0.5 : RBF 커널 (Radial Basis Function) 방사 기저 함수: 비선형

model.fit(X_train, y_train) # 훈련 데이터를 사용해 SVM 분류리를 훈련

predictions = model.predict(X_test) # 훈련된 모델을 사용하여 테스트 데이터에서 예측
score = metrics.accuracy_score(y_test, predictions)
print('정확도: {0:0.3f}'.format(score)) # 0.3f 소수셋째자리까지 vs 0.2f 소수둘째자리까지
```



















