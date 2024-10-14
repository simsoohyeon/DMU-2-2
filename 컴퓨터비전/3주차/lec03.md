## 2. 이미지 읽고 화면에 표시하기
### • 이미지 파일을 읽고 화면에 띄우기
```
• 데이터 불러오기: cv2.imread(‘파일명’)
• 데이터 화면에 띄우기: cv2.imshow(‘창이름’, 데이터명)
• 화면 멈추기: cv2.waitKey(0) – 키 입력을 기다리는 대기함수 (0; 무한대기)
• 모든 창 닫기: cv2.destroyAllWindows() – 키보드 입력 값 발생 시 창 종료됨
```

```
import cv2
import sys

img=cv2.imread("./soccer.jpg") # 이미지 파일을 읽어들여 이를 Numpy 배열 형식으로 반환 

if img is None: # 파일을 제대로 읽지 못하면 img가 None이 되어 sys.exit() 함수 사용해 종료
    sys.exit("파일을 찾을 수 없습니다.")

cv2.imshow("Image", img) # 첫 번째 인자는 창의 이름, 두 번째 인자는 이미지 데이터 

cv2.waitKey(0) # 키보드 입력을 대기하는 함수, 0은 무한정 대기로 키를 누를 때까지 창이 유
cv2.destroyAllWindows() # 사용자가 키보드 입력을 통해 창을 닫을 때 창 종료 
```
### • 이미지 데이터의 타입과 형태
```
• 이미지 타입: Numpy Array
• 이미지 형태: 3차원 데이터, (height = 행의 수, width = 열의 수, channels = 컬러정보)
• 채널이란? 이미지의 색상 정보를 표현하는 차원 (그레이스케일=1, 컬러=3)
• 컬러 채널은 blue, green, red 순으로 구성 (bgr)
```
```
import cv2
import sys

# 1. 이미지 파일 읽기
img = cv2.imread('soccer.jpg')

# 2. 파일 확인 및 오류 처리
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

# 3. 이미지 데이터 타입 출력
print(type(img)) # numpy 배열임을 확인
=> <class 'numpy.ndarray'>

# 4. 이미지의 형태 출력
print(img.shape) # 이미지의 배열 형태 출력
=> (height, width, channels)로 배열의 크기 나타냄, 크기와 채널 수 확인 
```
### - 영상 화소의 표현
### 1. 화소의 표현
```
이미지의 각 픽셀은 0에서 255까지의 정수값으로 표현, 8비트로 표현되며 0은 검은색 255는 흰색
그 사이의 값들은 회색조나 컬러이미지를 나타내는데 사용
컬러 이미지의 경우 각 픽셀은 3개의 값 BGR로 구성되어 있으며, 각각 파란색, 녹색, 빨간색 채널의 값
```
### 2. 이미지 인덱싱과 슬라이싱
```
- OpenCV에서 이미지는 numpy배열로 저장되므로, 배열에서 사용되는 인덱싱과 슬라이싱이 그대로 적용
- 인덱스를 사용해 특정 픽셀의 색상 값에 접근할 수 있고, 슬라이싱을 통해 부분 이미지를 추출 가능
```
### 1. 특정 픽셀 값 접근하는 코드 print(img[0, 0, 0], img[0, 0, 1], img[0, 0, 2])
```
- 이미지의 첫 행, 첫 열의 픽셀에서 파란색 채널의 값 출력
- 동일한 픽셀의 녹색 채널 값 출력
- 동일한 픽셀의 빨간 채널 값 출력
=> img[0,0] 픽셀의 BGR 값 각각 출력 
```
### 2. 다른 픽셀 값 접근 print(img[0, 1, 0], img[0, 1, 1], img[0, 1, 2])
```
첫 번째 행, 두 번째 열의 픽셀 값을 BGR 순서대로 출력 
```
### - 이미지 데이터 복사하기 -> 데이터.copy() 함수로 복사
```
import cv2

# 1. 이미지 파일 읽기
img = cv2.imread('soccer.jpg')

# 2. 이미지 데이터 복사, 깊은 복사
img2 = img.copy()

# 3. 동일한지 비교
print(img.all() == img2.all())
```
## 4. 이미지 형태와 크기 변환하기 
```
import cv2
import sys

# 이미지 파일 읽기
img = cv2.imread('soccer.jpg')

# 이미지가 없을 경우 종료
if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

# 이미지 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.cvtColor() 함수는 이미지를 다른 색상 공간으로 변환하는 함수
# cv2.COLOR.BGR2GRAY는 BGR 형식에서 그레이스스케일로 변환하는 옵션 

# 이미지 크기를 50% 축소
img_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# (0,0)은 새로운 크기를 직접 지정하지 않는다는 의미, fx와 fy를 사용해 축소 또는 확대 비율 지정

# 이미지 출력, cv2.imshow()는 이미지를 화면에 띄우는 함수
cv2.imshow('Image', img)
cv2.imshow('Gray Image', gray)
cv2.imshow('Small Image', img_small)

# 이미지 저장, cv2.imwrite()는 이미지를 파일로 저장하는 함수
cv2.imwrite('gray.jpg', gray)
cv2.imwrite('small.jpg', img_small)

# 화면 유지 및 모든 창 닫기
cv2.waitKey(0)
# 키 입력을 대기하는 함수로, 사용자가 키를 누를 때까지 창이 유지됨
cv2.destroyAllWindows()
# 모든 OpenCV 창으 닫는 함
```
### cvtColor 함수가 컬러 영상을 명암 영상으로 바꾸는 방법
### 아래 식을 이용해 데이터를 변환 (RGB 컬러 도메인 별 비율이 상이함, round는 반올림)

![image](https://github.com/user-attachments/assets/b485ad61-55a8-416f-a077-bd4fb14987f5)
```
이 공식은 RGB 값을 하나의 그레이스스케일 값으로 변환하는 과정
각 채널이 그레이스케일 값에 미치는 영향을 가중치를 적용하여 결정
이런 가중치는 인간의 시각이 녹색에 가장 민감하고, 파란색에 덜 민감하다는 사실 반영
```
### 1. 컬러이미지의 그레이스케일 변환은 가중합 방식을 사용해 RGB값을 하나의 값으로 변환
### 2. R,G,B채널의 기여도는 각각 29, 58, 11%로 인간의 시각에 따라 가중치가 다르게 적용
### 3. OpenCV의 cv2.cvtColor()함수는 이 과정을 자동화하여, BGR이미지를 그레이스케일로 변환
### 4. 결과적으로 컬러정보가 없는 명암 이미지가 생성되며, 이미지는 2차원 배열로 변

### OpenCV를 사용해 이미지의 그레이스케일 변환, 크기축소, 이미지 저장
#### 1. 컬러 이미지를 그레이스스케일로 변환: cv2.cvtColor()
```
gray = cv2.cvtColor(이미지데이터, 변환컬러)
# 컬러 이미지를 그레이스케일로 변환하려면, cv2.COLOR_BGR2GRAY
```
#### 2. 이미지 크기 변환 (축소, 확대): cv2.resize()
```
resized_img = cv2.resize(이미지데이터, (가로, 세로))  # 또는
resized_img = cv2.resize(이미지데이터, (0, 0), fx=비율x, fy=비율y)
# 크기를 직접 지정하지 않으려면 (0,0)으로 두고, 비율(fx,fy) 설정해 크기 변환
```
#### 3. 이미지 파일로 저장: cv2.imwrite()
```
cv2.imwrite('파일명', 이미지데이터)
# 파일명에는 저장할 확장자도 포함하여야 
```
### OpenCV 사용해 이미지 덮어쓰기, 차이 계산, 마스크 이미지 생성
### 이미지 간의 차이를 계산하고, 해당 차이를 이진화하여 마스크 이미지 
```
• 이미지 덮어쓰기, 원본과의 차이 확인, 마스크 이미지 생성하기
• 데이터[index] = 값 형태로 덮어쓰기 가능
• 이미지 차이 계산: cv2.absdiff(데이터1, 데이터2)
• 이미지 차이에 따른 이진화: cv2.threshod(차이, 기준값, 변환(흰색 255), 유형(이진화))
```
```
import cv2

# 이미지 파일 읽기 및 그레이스케일 변환
img = cv2.imread('soccer.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 덮어쓰기
img2 = img.copy()
# 이미지 특정 영역(50부터 100까지의 행과 열)을 선택하여 해당 영역을 255=흰색으로 덮어씀
# 이미지를 인덱싱하여 직접 값을 변경할 수 있음을 보여줌 
img2[50:100, 50:100] = 255


# 이미지 차이 계산: cv2.absdiff()
# 두 이미지 간의 절대 차이를 계산하는 함수, 원본과 수정된 이미지를 픽셀 차이를 계산해 저장
diff = cv2.absdiff(img, img2)

# 차이에 따른 이진화 처리: cv2.threshold()
# 이미지 차이를 기반으로 이진화 처리를 하는 함수
# 첫 번째 인자는 차이이미지, 두 번째 인자는 기준값 30, 이 값보다 큰 픽셀은 255흰색으로,
작은 값은 0 검은색으로 설정, cv2.THRESH_BINARY는 이진화방식 지정하는 옵션으로 기준값을 기준으로 흰검 변환
_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 이미지 출력: cv2.imshow() 이미지를 창에 표시하는 함수 
cv2.imshow('img', img)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)

# 화면 유지 및 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 마스크(임의생성)통해 배경 이미지와 전경 이미지 합성하기
```
• 마스크 데이터 생성: np.zeros(크기, 데이터타입) → 임의영역에 255 할당
• 이진이미지 반전: cv2.bitwise_not(이진이미지) → 255, 0 영역이 서로 반전됨
• 마스크 영역에 이미지 붙여넣기: cv2.bitwise_and(데이터, 붙여넣을 데이터, mask=영역)
• 이미지 더하기: cv2.add(데이터1, 데이터2)
```
```
import cv2
import numpy as np

# 배경 이미지 읽기 및 그레이스케일 변환
bg = cv2.imread('soccer.jpg')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# 전경 이미지(공) 읽기 및 그레이스케일 변환
ball = cv2.imread('soccer_ball.jpg')
ball = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
ball = cv2.resize(ball, (40, 40))

# 전경 이미지가 들어갈 fg 생성 및 공 이미지 넣기
# 1. 마스크 데이터 생성
# np.zeros(): 배경 이미지와 같은 크기의 검정색 0으로 채워진 빈이미지 생성
fg = np.zeros(bg.shape, dtype=np.uint8)
# 전경 이미지를 해당 영역에 덮어쓰기하여, 전경이미지를 특정 위치에 놓음
fg[100:140, 100:140] = ball

# 2. 마스크 이미지 생성
# mask: 전경 이미지가 위치할 영역을 255흰색으로 설정하고 나머지는 검은색으로 둠
mask = np.zeros(bg.shape, dtype=np.uint8)
mask[100:140, 100:140] = 255
# cv2.bitwise_not(mask): 마스크의 색을 반전하여 전경 이미지가 위치하지 않을 영역을 흰색,
전경 이미지 영역을 검정색으로 만듦, 반전된 마스크는 배경 이미지와 결합할 때 사용
mask_inv = cv2.bitwise_not(mask)

# 3. 배경과 전경 결합
bg = cv2.bitwise_and(bg, bg, mask=mask_inv)
# 배경 이미지와 마스크 결합
# 반전된 마스크 mask_inv사용해 전경 이미지가 위치할 영역을 비워둔 배경 이미지 만듦
fg = cv2.bitwise_and(fg, fg, mask=mask)
# 전경 이미지와 마스크 결합
# 원래 마스크를 사용하여 전경 이미지가 윛한 영역만 추출한 이미지 
result = cv2.add(bg, fg)
# 배경 이미지와 전경 이미지를  합쳐 최종적으로 이미지 결합 

# 이미지 출력
cv2.imshow('bg', bg)
cv2.imshow('fg', fg)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
```
![image](https://github.com/user-attachments/assets/050728e6-26a5-47c3-bd64-b1f50f4cf259)

## 4. 비디오 읽기
```
• 웹 캠 열기: cv2.VideoCapture(0, cv2.CAP_DSHOW)
• 반복문을 통해, 연속 영상을 입력: ret, frame = cap.read() 
```
```
import cv2
import sys

# 1. 웹캠 열기: cv2.VideoCapture()
# 첫 번째 인자는 웹캠 장치, 0은 기존 카메라 의미
# cv2.CAP_DSHOW 윈도우에서 비디오 캡처를 안정적으로 처리하기 위한 옵션 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 2. 웹캠이 정상적으로 열렸는지 확인: cap.isOpened()
if not cap.isOpened():
    sys.exit("카메라를 열 수 없습니다.")

# 3. 실시간 비디오 프레임 읽기 및 출력: cap.read()
# ret은 프레임을 읽는데 성공했는지 여부, frame은 읽어들인 이미지 데이터 담음
# ret이 False일 경우 프레임을 읽는데 문제가 -> 반복문 종료
while True:
    ret, frame = cap.read()
    if not ret:
        print("비디오 읽기 오류")
        break
    
    # 4. 비디오 프레임 출력: cv2.imshow()
    cv2.imshow('video display', frame)
    
    # 5. 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 6. 리소스 해제 및 창 닫기
cap.release() # 웹캠 장치 해
cv2.destroyAllWindows()
```
### 저장된 비디오 파일 재생하기
```
import cv2
import sys

# 1. 비디오 파일 열기: cv2.VideoCapture()
# 비디오 파일을 열기위한 함수, 인자로 비디오 파일의 경로 전달
cap = cv2.VideoCapture('movingobj01.mp4')

# 2. 비디오 파일이 정상적으로 열렸는지 확인: cap.isOpened()
if not cap.isOpened():
    sys.exit('비디오 파일을 열 수 없습니다.')

# 비디오 파일 재생
while True:
    ret, frame = cap.read() # 3. 비디오 프레임 읽기: cap.read()
    if not ret:
        break
    
    # 4. 프레임을 화면에 출력
    cv2.imshow('video display', frame)
    
    # 'q' 키를 입력하면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 비디오 파일을 닫고 창을 닫음
cap.release()
cv2.destroyAllWindows()
```
### N초마다 한 번씩 프레임 캡처하여, 가로로 이어 붙이기 
```
• 초당 촬영된 이미지 수 불러오기(fps;frame per second): fps = cap.get(cv2.CAP_PROP_FPS)
• 이미지 가로로 이어붙이기: cv2.hconcat(연속데이터, h;horizontal, 세로 cv2.vconcat, v;vertical)
```
```
import cv2
import sys

# 비디오 파일 열기
cap = cv2.VideoCapture('movingobj01.mp4')

# 비디오 파일이 열리지 않으면 종료
if not cap.isOpened():
    sys.exit('비디오 파일을 열 수 없습니다.')

# 1. 초당 프레임 수(fps) 가져오기: cap.get()
# cap.get()은 비디오 속성 값을 가져오는 함수
# cv2.CAP_PROP_FPS는 초당 프레임 수를 의미
# 비디오 당 초당 몇 개의 프레임이 있는지 계산하고, 이 값을 기준으로 프레임 추출 간격 설정 
fps = cap.get(cv2.CAP_PROP_FPS)

# 2. 프레임 추출을 위한 반복문 
count = 0
frames = []
n = 1  # N초마다 프레임을 추출하는 간격 설정 (여기서는 1초)

# 비디오 파일에서 프레임 추출
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # N초마다 프레임 저장
    # n초마다 한 번씩 프레임을 추출하는 로직, n은 추출간격 = 초, fps*n은 프레임 수
    # 프레임 저장: 추출한 프레임은 frames 리스트에 저장 
    if count % int(fps * n) == 0:
        frames.append(frame)
    count += 1

# 3. 프레임들을 가로로 이어붙이기: cv2.hconcat()
# 인자로 들어가는 리스트 frames안의 이미지들이 가로로 결합, 세로는 cv2.vconcat()
result = cv2.hconcat(frames)

# 결과 이미지 출력
cv2.imshow('result', result)

# 종료 조건 설정
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
```
### 연속적으로 이어붙인 데이터의 shape
```
OpenCV로 이어붙인 이미지데이터의 shape 차원을 설명
여러 프레임을 가로로 이어붙인 후, 각각의 프레임과 결과 이미지의 shape을 확인하는 과정 
이미지의 배열 구조와 이어붙인 후의 차원 변화 이해 
```
![image](https://github.com/user-attachments/assets/ffbecb1c-df91-4fde-a017-5b1137cdf64c)













































