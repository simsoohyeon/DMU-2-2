<div align="center">
  <h2>2장-3장 SVM (Support Vector Machine)</h2>
</div>

## 🟣 SVM의 정의
```
데이터를 분류할 때 두 클래스 간의 경계를 설정하고, 그 경계를 중심으로 최대한 큰 마진 확보하여 데이터 분류하는 방식
```
### 1. 이진분류
```
SVM은 두 개의 클래스로 데이터를 분류하는 이진 분류 문제에 사용
ex) 빨간색과 파란색 점들처럼 두 그룹으로 나누는 경우
```
### 2. 마진의 최대화
```
- SVM은 각 클래스 사이의 마진(두 클래스 사이의 여유 공간)을 최대화하는 것으로 목표
그래프에서 검은 선은 하이퍼플레인, 두 클래스를 나누는 기준이며 이 기준선에서 각 클래스로부터
가장 가까운 데이터 포인트들(서포트 벡터라고 불리는 데이터)가 마진 결정
- 서포트 벡터는 하이퍼플레인 가까이에 위치한 데이터 포인트들로, 이들이 마진을 정의, 분류 경계 형성
```
## 🟣 이상값 Outliers이 SVM의 분류에 영향을 미치는 영향
### 1. Hard Margin, C값이 높은 경우
```
- C값이 높은 경우: C값이 클 때 SVM이 오류(오분류)를 거의 허용하지 않고 매우 엄격하게 분류할 때
- 하이퍼플레인: 빨간색과 파란색 데이터 포인트를 완벽하게 분리하기 위해 매우 좁은 마진을 형성,
모든 데이터를 완벽히 구분하려고 함
- 이상값의 영향: 그래프 상단에 빨간색 이상값은 이 하이퍼플레인은 크게 왜곡시키고 있으며,
이로 인해 일반적인 데이터 포인트에 비해 성능이 떨어질 수 있음
SVM이 엄격하게 동작하므로, 이상값이 큰 영향 미침
```
### 2. Soft Margin, C값이 낮은 경우
```
- C값이 낮은 경우: C값이 낮을 때 일부 오류를 허용하고 데이터가 완벽히 분리되지 않아도 되는 경우 보여줌
- 하이퍼플레인: 마진이 넓어졌으며, 몇몇 빨간색 점과 파란색 점들이 마진 안으로 들어와있음
=> 즉 일부 데이터를 잘못 분류해도 괜찮다는 의미
- 이상값의 영향: 이상값이 존재하더라도 하이퍼플레인은 크게 왜곡되지 않고, 더 넓은 마진 유지, 영향 덜 받음
```
### 비교분석
```
- 하드마진, C값이 높을 떄: 이상값을 고려하지 않고 모든 데이터를 분류하려고 하지만, 이상값의 영향으로 성능 저하
- 모델이 오류를 허용하지 않도록 강하게 규제 -> 모든 데이터를 올바르게 분류하려고 하며, 오분류 데이터 최소화
- 마진이 좁아지고 하이퍼플레인이 데이터 포인트에 가까워짐 
- 노이즈가 있는 경우에도 데이터를 완벽하게 분류하려고 하다보니, 모델이 지나치게 복잡해져서 과적합(Overfitting)이 발생

- 소프트마진, C값이 낮을 때: 이상값이 있더라도 조금의 오류를 허용하면서 더 넓은 마진을 유지하므로, 이상값의 영향 덜 받음
- 모델이 오류를 허용하며, 일부 데이터를 잘못 분류해도 괜찮다고 판단
- 마진이 넓어지고, 이상값에 덜 민감해지면서 더 일반적인 패턴 학습
- C값이 너무 작으면 모델이 과소적합(Underfitting)이 될 수 있으며, 데이터의 세부적인 패턴 놓침
```
![image](https://github.com/user-attachments/assets/a3b0b5c3-5ba8-41f3-8661-2d4d76981d5d)

<div align="center">
  <h2>SVM 1</h2>
</div>

```
from sklearn.svm import SVC
classifier = SVC(kernel="linear") # 선형커널을 사용한다는 의미

# 2차원 평면에 위치한 데이터 포인트들, x,y 형태로 표현
training_points = [[1, 2], [1, 5], [2, 2], [7, 5], [9, 4], [8, 2]]
labels = [1, 1, 1, 0, 0, 0]
# SVM 모델 학습, 데이터 포인트에 맞는 최적의 하이퍼플레인 찾
clasffier.fit(training_points, labels)

# SVM은 여러 개의 데이터 포인트를 동시에 예측, 행렬(2차원 배열) 형태로 데이터 받아들임 [[ ? ]]
print(classifier.predict([[3, 2]]))

# 서포트 벡터 출력, 하이퍼플레인에 가장 가까이 위치한 데이터 포인트들
# 실제 분류 경계를 결정하는데 중요한 역할 
print(classifier.support_vectors_)
```

<div align="center">
  <h2>SVM 2</h2>
</div>

```
# Read data and split data on 8:2 ratio
# read_data 함수는 두 파일에서 데이터를 읽고, 두 클래스 좌표와 해당 레이블 반환
x, labels = read_data("points_class_0.txt", "points_class_1.txt")
# 훈련:테스트 비율 => 80:20
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size = 0.2, random_state=0) 
print("Displaying data. Close window to continue.")
# plot_data 함수는 훈련 데이터와 테스트 데이터를 시각화하여 2차원 평면에 표시 
plot_data(X_train, y_train, X_test, y_test)
# SVC는 선형 SVM 분류기 생성
# kernel="linear" 선형 커널을 사용하여, c=1는 정규화 상수로, 마진을 넓게 유지하며 오류 허용하는 방
clf = svm.SVC(kernel='linear', C=1)
clf_1.fit(X_train, y_train)
print("Display decision function (C=1) ...\n The SVM classifier will choose a large margin decision boundary at the expense of larger number of misclassifications") 
# 결정경계 시각화 
plot_decision_function(X_train, y_train, X_test, y_test, clf_1) 
clf_100 = svm.SVC(kernel='linear', C=100) 
clf_100.fit(X_train, y_train) 
print("Accuracy(C=1): {}%".format(clf_1.score(X_test, y_test) * 100 )) 
print("\n") 
print("Display decision function (C=100) ...\nThe classifier will choose a low margin decision boundary and try to minimize the misclassifications") 
plot_decision_function(X_train, y_train, X_test, y_test, clf_100) 
print("Accuracy(C=100): {}%".format(clf_100.score(X_test, y_test) * 100 )) 
clf_1_predictions = clf_1.predict(X_test) 
clf_100_predictions = clf_100.predict(X_test)
```
![image](https://github.com/user-attachments/assets/a09c581c-1af5-4991-b4df-e96a399a7ab2)

### 🟣 정확도 출력 코드
```
print("Accuracy(C=1): {}%".format(clf_1.score(X_test, y_test) * 100))
```
```
- {}%.format(): 문자열 포맷을 위한 메소드, format() 함수에 의해 값으로 대체
- clf_1.score(): SVM모델의 정확도를 계산하는 함수, 테스트 데이터에 대해 올바르게 분류한 비율 반환

➕ ➕ ➕ ➕ ➕
- {:.4f} {위치 인덱스 지정, 포맷팅 값 여러 개 있을 때 어디 사용할지:소수점 형식으로 소수점 아래 몇까지 표시}
```
## 🟣 그래프 출력 결과에 따른 C값 선택 
![image](https://github.com/user-attachments/assets/201e9270-6288-4e61-b111-815ce19f78ed)
### - C=1일 때
```
- training data 분류: 부정확 (훈련 데이터에 대한 정확도가 낮을 수 있음)
- margin: 크다, 마진을 넓게 유지하면서 일부 오분류 허용
- 사용: noise가 많은 데이터셋, 이상값이 많을 때 적절한 선택
```
### - C=100일 때
```
- training data 분류: 정확 (훈련 데이터에 대한 정확도가 높음)
- margin: 작다, 매우 좁아지고, 훈련 데이터에 과도하게 맞추는 경향
- 사용: noise가 적은 데이터셋, 잡음이 적고, 데이터가 잘 정돈된 상황에서 사
```


















