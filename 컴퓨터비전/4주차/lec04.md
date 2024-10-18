<div align="center">
  <h1> 컴퓨터 비전 </h1>
    <h1> 4주차 </h1>
</div>

## <영상처리 Image Processing>
```
정의: 특정 목적을 달성하기 위해 원래의 영상을 개선하거나 새로운 영상으로 변환하는 작업
목적: 영상의 품질을 개선하거나, 특정한 정보 추출하는데 사용
```
## 영상처리 기술 사용 예시
### 1. 화질 개선 목적
```
- super resolution: 해상도가 낮은 이미지를 고해상도로 변환해 화질 개선하는 기술
- noise removal: 촬영된 이미지나 영상에 존재하는 잡음을 제거하는 기술
- deblur: 카메라의 흔들림으로 인한 블러 제거해 선명하게 만드는 기술
```
### 2. 구직 영상 화질 개선
```
- 오래된 영상이나 무한도전과 같은 구작 영상의 화질을 개선해 시청 품질을 향상시킴
- 촬영된 이미지에서 노이즈 제거, 흔들린 영상 개선, 그림자/안개 제거 등 작업 통해 더 나은 영상 제공
```
### 3. 컴퓨터 비전 기술 개발
```
전처리 과정 Pre-processing: 컴퓨터 비전 기술을 개발할 때, 영상처리는 이미지나 영상을 분석하기 전의 전처리 과정
이를 통해 이후 분석을 위한 품질 좋은 데이터로 변환 
```
### 4. 영상 처리 응용
```
- 이미지 인식: 영상처리 기술을 통해 이미지를 분석하여 자동차 번호판 식별, 의료영상
=> 다양한 분야에서 활용되며, 그 목적에 따라 기술을 맞춰 적용할 수 있음 
```
![image](https://github.com/user-attachments/assets/a85d28bc-10b7-456c-8227-66a26465530e)
## 1. 영상 획득과 표현
### - 우리가 보는 영상 데이터
```
- 디지털 디바이스
현재 사회에서 우리가 접하는 영상 데이터는 디지털 장치를 통해 감상,
아날로그 신호가 아닌 디지털 신호로 변환된 영상을 처리하고 표시
```
### - 디지털 영상
```
- 아날로그 신호와 디지털 신호
- 영상은 원래 아날로그 신호로 연속적으로 존재함, 아날로그 신호는 곡선 형태로 시간에 따라 부드럽게 변화
- 아날로그 신호를 디지털화 discreate하여, 즉 시간적으로 끊어진 신호를 변환하여 표현한 것이 디지털 영상
디지털 신호는 일정한 간격의 직선으로 표현
- 디지털 영상은 아날로그 신호를 샘플링하고 양자화하여 만들어짐 -> 컴퓨터나 디지털 장치에서 처리가 가능
```
### - 디지털 영상 활용 분야
```
- 영상감상
- 컴퓨터 비전 기술
: 디지털 영상은 컴퓨터 비전 기술을 개발할 때, 핵심적 요소로, 컴퓨터가 영상을 분석하고 처리할 수 있는 기반
ex) 물체인식, 얼굴 인식, 자율주행 등 다양한 분야 
```
![image](https://github.com/user-attachments/assets/893f23f3-4396-4f61-84b1-95a0b63f838d)

### - 디지털 변환
#### 1. 샘플링 Sampling 
```
- 정의: 샘플링은 아날로그 영상을 디지털 영상으로 변환하기 위한 첫 번째 단계
이 과정에서 영상은 M*N 크기의 격차로 나누어지고, 각 격자마다 해당 영역의 빛 강도를 측정하여 수치화
- 목적: 아날로그 영상은 연속적인 신호로 표현되는데, 이를 일정한 간격으로 나누어 불연속적인
데이터로 바꾸는 과정이 필요함 이때 나눈 격자의 크기에 따라 해상도가 결정되며, 더 작은 격자로
나누면 더 높은 해상도를 얻을 수 있음
```
#### 2. 양자화 Quantization
```
- 정의: 샘플링된 데이터를 수치화하여 디지털 영상으로 변환하는 과정
샘플링을 통해 얻은 값(빛의 강도)를 L단계로 나누어 정해진 값으로 변환하는 것이 양자화
- 목적: 샘플링을 통해 얻은 각 격자의 밝기 값을 실제로 사용 가능한 디지털 수치로 변환하는 과정
아날로그 신호가 디지털 값으로 변환되며, 이 값은 보통 픽셀 값으로 나타남
- 양자화 단계: 이미지의 픽셀값은 보통 0~255 사이의 값으로 표
```
#### <디지털 영상의 해상도 표준>
```
- 해상도 표준은 가로 및 세로 픽셀 수를 기준으로 디지털 영상의 해상도를 나타냄
- 샘플링 수준에 따라 화소 수가 결정되며, 가로와 세로의 픽셀 수가 많을 수록 더 높은 해상도를 가진 영상이 됨
즉, 해상도는 화면을 구성하는 픽셀의 수 의미, 더 많은 픽셀이 존재할 수록 세밀한 영상 표현
```
```
• 디지털 변환
1. M*N 으로 샘플링 (sampling): 아날로그 영상을 디지털화 하기 위해, 일정 간격으로 나누어 수치화
2. L 단계로 양자화 quantization: 샘플링을 통해 얻은 수치를 다시 정해진 수준으로 변환
```
![image](https://github.com/user-attachments/assets/f11d0e8b-db5c-4f7a-9ccc-bc02ea12eef3)

### 영상 내 색상의 깊이를 표현하는 색 심도 Color Depth
```
색심도: 이미지를 구성하는 픽셀 당 표현할 수 있는 컬러 정보의 양을 의미
한 픽셀이 표현할 수 있는 비트 수에 따라 그 픽셀이 표현할 수 있는 범위가 달라짐
더 많은 비트를 사용할수록 더 많은 색상을 표현할 수 있어, 화질이 풍부해
```
![image](https://github.com/user-attachments/assets/d5ece3f2-7248-4890-9eef-a4c91d25a77c)
### 주사율 Refresh Rate
```
- 주사율은 동영상이나 디스플레이에서 초당 화면이 갱신되는 횟수를 나타내는 수치
- 보통 헤르츠 Hz로 표현되며, 주사율이 더 높을수록 더 부드러운 화면 전환이 가능
ex) 60Hz 주사율을 가진 디스플레이는 1초에 60번 화면 갱
```
### 다양한 종류의 영상 데이터
```
a 명암 영상: 2차원 배열(텐서)로 표현된 단일 채널 영상, 주로 흑백 이미지
b 컬러 영상: 3차원 텐서로 표현된 영상, RGB 채널을 포함해 색상 나타냄
c 컬러 영상: 4차원 텐서로 표현되며, 시간 축이 추가되어 동영상 나타냄
d 다분광/초분광/MR/CT영상: 다양한 스펙트럼 또는 의료 이미지 사용되는 고차원 영상
e RGB-D영상: 깊이 정보를 포함한 컬러 영상, 주로 3D스캔이나 깊이 측정하는데에
f 점구름 영상 point cloud: 라이더나 3D스캐너로 얻은 3차원 좌표계에서의 점들의 집합, 3D 구조 표
```
![image](https://github.com/user-attachments/assets/cb17964c-8519-4cf0-bc5b-35b4dd32465a)

## 영상 색 공간
### 색의 표현 및 재현 방식을 정의하는 색 공간 
```
1.🔴 RGB 색공간: 빨강, 녹색, 파랑 세 가지 기본 색을 조합하여 색을 표현하는 방식
=> 모니터, 티비, 카메라 등 다양한 디스플레이 장치에서 주로 사용 
2. CMYK 색공간: 네 가지 색을 혼합하여 색을 표현하는 방식, 인쇄에 주로 사용
3.🔴 HSV 색공간: 색상, 채도, 명도로 색을 표현하는 방식, 색의 직관적인 조작 -> 그래픽, 디자
4.🔴LAB 컬러모델: CIE 색 공간의 일부로, 모든 인간이 볼 수 있는 색상을 수학적으로 표현
L은 밝기를 a는 Red/Green, b는 Blue/Yellow 값을 표현
색상의 절대적인 값을 나타내므로, 다양한 디스플레이 및 출력장치 간의 색상 일관성 표현
5. CIE 색공간: 국제조명위원회가 정의한 색공간, 모든 가시광선을 포함해 색을 수치적으로 표현
6. CIELAB 색공간: 세 축을 통해 색을 표현하는 방식, 색 차이 정밀하게 계산하여 색상 분석에 사용 
```
![image](https://github.com/user-attachments/assets/19815139-bd49-4946-befd-52443e1cd4e8)

### numpy의 슬라이싱 기능을 이용해 RGB 채널별로 디스플레이
```
1. RGB 채널
- 이미지의 각 픽셀은 RGB 채널로 구성, 빨강, 초록, 파랑의 세 가지 색으로 표현
- 각 채널은 2차원 배열의 값으로 표현되며, 해당 배열을 슬라이싱하여 각 채널을 개별적으로 추출
2. 슬라이싱을 통한 채널 분리
- 슬라이싱: numpy.ndarray배열에서 특정 범위를 잘라내는 기능이용해 채널을 각각 분리
ex) img[:,:,0]: 이미지의 파란색 채널만을 추출
    img[:,:,1]: 이미지의 초록색 채널만을 추출
    img[:,:,2]: 이미지의 빨간색 채널만을 추출
ex) 일부 영역 슬라이싱 img[0:img.shape[0]//2, 0:img.shape[1]//2, :]
- 세로방향: 이미지의 상단 절반 선택(이미지 행의 절반)
- 가로방향: 이미지의 왼쪽 절반선택(이미지 너비 절반)
- 채널: RGB 모든 채널 포 

```
![image](https://github.com/user-attachments/assets/a1142ba7-0254-406e-ab1d-f2b494e98828)

```
import cv2

# 이미지 읽기
# 이 변수는 numpy 배열 형식으로, 이미지에 픽셀 데이터 포
img = cv2.imread("soccer.jpg")

# R,G,B 채널 분리, 기본적으로 BGR순서로 저
B,G,R = cv2.split(image)

# 각각의 채널을 별도로 보여주기
cv2.imshow('orginal', image)
cv2.imshow('red', R)
cv2.imshow('green', G)
cv2.imshow('b',B)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destoryAllWindows()
```
### RGB 채널별 밝게 보여지는 영역이 다르게 나타
![image](https://github.com/user-attachments/assets/a1a3455e-793c-42b4-9958-d6f00fd75921)

## 3. 영상 필터링
### 단순 덧셈 기반의 밝기 보정
```
1. cv2.add() 함수
- cv2.add(이미지, (100))은 이미지의 각 픽셀 값에 100 더하는 연산 수행
- 이 함수는 이미지를 밝게 하기 위해 사용, 추가되는 값에 따라 이미지의 밝기가 증가
2. 밝기 보정
- 이미지의 픽셀값은 0~255로 표현, 각 픽셀 밝기 값에 100을 더하면 픽셀값이 더 높아져 이미지가 밝게 보임
- but 255를 넘으면 최대값인 255로 포화되어 나타남
이미지에서 더 이상 밝은 부분의 차이가 표현되지 않으며, 흰색으로 보임
3. 결과 이미지
- 원본 이미지는 자연스러운 밝기 유지, 수정 이미지는 전체적으로 발가지고, 밝은 영역에서 흐려진 부분 생김
(픽셀 값이 255 넘어가는 부분이 많아졌기 때문에, 명암 차이가 사라지고 하얗게 표현된 것)
```
![image](https://github.com/user-attachments/assets/f66355d8-c484-4eed-8dd4-012570640664)

### OpenCV를 활용한 밝기 보정 convertScaleAbs
```
import cv2 # 1. openCV 라이브러리 불러오기, 이미지 처리 작업 수행하는데 사용
image = cv2.imread('food.jpg') # 이미지는 넘파이 배열로 저
# 대비와 밝기 조정 (대비 1.2배, 밝기 +30)
alpha = 1.2 # 대비 계수 (1보다 크면 대비 증가, 1보다 작으면 대비 감소)
beta = 30 # 밝기 증가 값 (양수면 밝기 증가, 음수면 밝기 감소)
# 대비와 밝기 조정 적용
adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imshow('Original Image', image)
cv2.imshow('Adjusted Image (Contrast & Brightness)', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
```
- cv2.convertScaleAbs(): new value= alpha*old_value+beta로 계산
=> alpha: 픽셀 값에 곱해져 대비를 조절, beta: 픽셀 값에 더해져 밝기를 조절
- alpha는 대비를 조정하는 계수, 1보다 크면 대비가 증가하고, 1보다 작으면 대비가 감소
- beta는 밝기를 조정하는 값, 양수이면 밝기가 증가하고, 음수이면 밝기가 감소
```
### 합성곱 Convolutional 연산을 통한 영상 필터링
#### 스무딩: 영상을 부드럽게 blur, 샤프닝: 영상을 날카롭게, 엠보싱: 영상이 울퉁불퉁하게
```
1. 스무딩 필터 Blurring Filter: 이미지를 부드럽게 만드는 필터
- 필터의 요소가 모두 1/9로 설정되어, 주위 픽셀 값을 평균내어 부드럽게 만드는 역할
- 이미지에서 노이즈를 제거하거나 경계선을 흐리게 만들어 부드러운 효과
2. 샤프닝 필터 Sharpening Filter: 이미지를 선명하게 만드는 필터
- 중앙 값이 높고, 주변 값이 음수인 필터 구조 가짐 -> 이미지의 중앙 부분 강조, 이미지 선명
3. 엠보싱 필터 Embossing Filter: 이미지에 입체감을 주는 필터
- 엠보싱 필터는 픽셀 값 차이를 이용해 윤곽선을 강조하고, 입체적인 효과
- 음수와 양수의 값이 적절히 배치되어, 경계 부분이 더 강조되고 이미지가 볼록 or 오
```
![image](https://github.com/user-attachments/assets/7239c4a9-88b5-4c29-8aac-5d86fd655dd5)

### 합성곱 Convolutional 연산 적용 방법
```
- 1차원 컨볼루션은 1차원 배열에 커널을 적용해 새로운 값을 계산하는 방식
- 2차원 컨볼루션은 2차원 배열에서 필터가 각 픽셀에 적용되며, 이미지의 세부적인 패턴 추출 or 변
```
![image](https://github.com/user-attachments/assets/c3acea4b-5339-4c68-97f8-bbe5ec755cd4)

### 이미지에 샤프닝 필터 적용 
```
import cv2
import numpy as np
image = cv2.imread('food.jpg')
# 3x3 샤프닝 커널 정의
# 중앙 값이 양수, 주변 값이 음수로 설정되어 가장자리를 강조함, 값이 클수록 경계가 더 뚜렷 
sharpen_kernel = np.array([[ 0, -1, 0],
                          [-1, 4, -1],
                          [ 0, -1, 0]])
# 샤프닝 필터 적용 cv2.filter2D() 이미지에 커널(필터) 적용
# 첫 번째 인자는 원본 이미지, 두 번째 인자는 출력 이미지의 깊이(-1로 설정하면 원본
이미지와 동일한 깊이), 세 번째 인자는 적용할 커널
sharpened_image1 = cv2.filter2D(image, -1, sharpen_kernel)

# 커널 중앙 값 변경 및 추가 필터 적용
# 중앙값을 5로 변경한 후, 다시 필터를 적용 -> 값이 클수록 더 강하게 샤프닝
sharpen_kernel[1,1] = 5
sharpened_image2 = cv2.filter2D(image, -1, sharpen_kernel)

# 중앙값 9로 변경 + 필터링
sharpen_kernel[1,1] = 9
sharpened_image3 = cv2.filter2D(image, -1, sharpen_kernel)

# 결과 시각화
# cv2.hconcat() 여러 이미지를 수평으로 결합하여 한 이미지로 만듦 (리스트가 인)
result = cv2.hconcat([image, sharpened_image1, sharpened_image2, sharpened_image3])
cv2.imshow('Sharpening Results', result)
# 키 입력 대기
cv2.waitKey(0)
cv2.destroyAllWindows()
```
![image](https://github.com/user-attachments/assets/f483b60f-a53a-4ebd-9701-956d4c55d6fb)


### 영상, 박스필터, 가우시안 블러, 샤프닝, 엠보싱 순
```
import cv2 # 이미지 처리 작업 수행
import numpy as np # 배열 연산 수행 
image = cv2.imread('food.jpg') # numpy 배열저장, 각 픽셀을 BGR 값 포함

# 박스 필터 Box Filter, 평균 블러링
# cv2.blur()는 평균 블러링 함수로, 커널크기 (9,9)적용하여 이미지의 주변 픽셀 값을 평균내어 부드럽게
box_filtered = cv2.blur(image, (9, 9))

# 가우시안 블러링 Gaussian Blurring
# cv2.GaussianBlur() 가우시안 블러적용, 여기서 커널크기 9,9와 표준편차 10.0 설정
# 가우시안 블러는 평균 블러와 달리 가우시안 분포에 기반하여 주변 픽셀을 부드럽게 처리
gaussian_blurred = cv2.GaussianBlur(image, (9, 9), 10.0)

# 이미지 샤프닝 Sharpening
# cv2.addWeighted() 두 이미지의 가중치를 조합하는 함수
# img와 gaussian_blurred를 각각 1.5와 -0.5로 조합하여 이미지의 경계 강조하는 샤프닝 효과
# 원본이미지에 가우시안 블러를 빼서 션명한 이미지를 얻는 방식
sharpened_image = cv2.addWeighted(image, 1.5, gaussian_blurred, -0.5, 0)

# 이미지를 그레이스케일로 변환
# cv2.cvtColor()는 컬러 이미지를 그레이스케일로 변환하는 함수
# 이미지 처리에서 색상 정보를 제거하고 밝기 정보만을 사용해 수행하기 위한 전처리 단계
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 엠보싱 필터 적용 Embossing Filter: 입체감을 표현하는 필터
# emboss_kernel은 3*3행렬로 각 픽셀의 차이를 강조하여 입체 효과를 냄
# cv2.filter2D() 함수는 이 커널을 적용하여 필터링된 이미지 만듦
# np.clip()함수는 결과값에 128을 더한 후, 0~255 사이로 값 제한하여 적절한 밝기 범위로 만듦
# -1은 입력 이미지와 동일한 깊이를 유지하라는 뜻, 데이터 타입을 그대로 유지하며 필터링 작업
# 이미지의 깊이는 각 픽셀이 차지하는 비트 수, 이는 픽셀의 색상 값을 표현하는데 사용
# 일반적인 그레이스스케일 이미지는 8비트로, 픽셀값이 0~255 사이의 값 가짐 
emboss_kernel = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
embossed = cv2.filter2D(gray_image, -1, emboss_kernel)
embossed = np.uint8(np.clip(embossed + 128, 0, 255))

# 결과 이미지를 수평으로 결합
# cv2.hconcat() 함수는 여러 이미지를 수평으로 결합해 하나의 이미지로 만듦
# 엠보싱 이미지는 그레이스케일이므로, 이를 컬러로 변환하기 위해 cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
result = cv2.hconcat([image, box_filtered, gaussian_blurred, sharpened_image,
cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)])

cv2.imshow('Image Processing Results', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 주요 문법
```
1. cv2.blur(image, (9,9))
: 평균 블러링 적용하는 함수로, (9,9)크기의 커널을 사용해 이미지를 부드럽게 함

2. cv2.GaussianBlur(image, (9,9), 10.0)
: 가우시안 블러링 적용 함수, 커널 크기 9,9와 표준편차 10.0 사용해 이미지의 경계선 부드럽게 함

3. cv2.addWeighted()
: 두 이미지를 가중합하여 결합하는 함수 첫 번째 이미지는 alpha가중치를, 두 번재 이미지는 beta 가중치 부여받음

3. cv2.filter2D()
: 이미지에 필터를 적용하는 함수, 필터는 커널을 의미하며 커널은 이미지의 특정 패턴을 강조하거나 제거하는데 사용

4. np.clip()
: 주어진 배열의 값을 특정 범위 내로 제한하는 함수
엠보싱 필터 적용 후 결과값이 벗어나는 경우 이를 0~255사이의 값으로 제한 
```
![image](https://github.com/user-attachments/assets/b4875ecd-cbf9-4ef2-9952-ecf159fef770)

### 노이즈 추가 기법
```
• 소금후추(Salt and Pepper) 잡음: 검/흰 픽셀을 추가
• 가우시안 노이즈(Gaussian Noise): 각 픽셀에 가우시안 분포에 따른 랜덤 노이즈를 더하는 방식
```
#### - 소금후추 노이즈 Salt and Pepper Noise
```
이미지에 랜덤학 검은색 또는 흰색 픽셀이 추가된 노이즈
주로 디지털 이미지 전송 중 발생하는 신호 손실이나 잡음을 표현할 때 사용
- 이미지의 일부 픽셀이 임의로 흰색 또는 검은색으로 변경
- 노이즈가 발생한 부분은 극명한 흰색 점으로 표시, 다른 부분은 그대로 유지
- 소금 후추 노이즈는 특정 환경에서 발생할 수 있는 데이터 손실이나 잡음의 일종종
```
#### - 가우시안 노이즈 Gaussian Noise
```
이미지의 각 픽셀 값에 가우시안 분포에 따라 랜덤 값을 추가하는 노이즈
자연스러운 랜덤 잡음을 표현할 때 사용
```










