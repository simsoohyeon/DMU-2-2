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
- 가우시안 분포는 평균을 중심으로 한 대칭적인 분포로, 각 픽셀에 다양한 정도의 노이즈가 추가
- 가우시안 분포는 사진 촬영 시 센서 잡음이나 환경에서 발생하는 랜덤 잡음을 묘사
- 노이즈가 더해진 후 픽셀 값은 원본 값에 비해 조금씩 달라지며, 전체적으로 이미지 흐릿해질 수
```
### 소금후추 노이즈 Salt and Pepper Noise 이미지에 추가하기
```
import cv2 # 이미지 처리 작업 수행
import numpy as np # 배열 생성 및 난수 생성 처리
image = cv2.imread('soccer.jpg’) # 이미지 읽어서 배열 형식으로 반환

# 1. 후추소금 노이즈 (Salt and Pepper Noise) 적용
# img.copy() 원본 이미지를 손ㅅ낭시키지 않기 위해 img의 복사본을 만듦
# black = 0 픽셀 값을 검은색으로 설정, white 픽셀 값을 흰색으로 설정
# 이미지의 크기와 동일한 크기의 랜덤 확률 배열 생성, 0과 1 사이의 값 가짐
def add_salt_and_pepper_noise(img, prob): # 확률에 따라 랜덤하게 픽셀을 흑백(0, 255)으로 설정
noisy = img.copy()
black = 0
white = 255
probs = np.random.rand(img.shape[0], img.shape[1])
noisy[probs < prob] = black # prob보다 작은 픽셀을 검은색 0으로 설정
noisy[probs > 1 - prob] = white # 1-prob보다 큰 픽셀을 흰색 255로 설정 
return noisy

# 노이즈 적용, 이미지에 2% 확률로 노이즈 적용
# 0.02인 픽셀은 검은색, 1-0.02=0.98 확률을 가진 픽셀은 흰색으로
salt_pepper_noise_img = add_salt_and_pepper_noise(image, 0.02) # 2% 확률로 노이즈 추가
cv2.imshow('Original Image', image)
cv2.imshow('Salt and Pepper Noise', salt_pepper_noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 가우시안 노이즈 Gaussian Noise를 추가하는 이미지
```
import cv2
import numpy as np

image = cv2.imread('soccer.jpg’)

# 2. 가우시안 노이즈 (Gaussian Noise) 적용
# mean=0 가우시안 노이즈의 평균값으로 기본 값 0
# sigma=25 가우시안 분포의 표준편차로, 노이즈의 세기 결정
def add_gaussian_noise(img, mean=0, sigma=25):
row, col, ch = img.shape # 이미지의 (행,열,채널) 반환
# 가우시안 노이즈 생성
# 가우시안 분포에 따른 노이즈를 생성, 이 분포는 평균값 mean과 sigma기반으로 
gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
noisy_image = img + gauss.astype(np.uint8) # 원본 이미지에 더해져 노이즈가 추가된 이미지 만듦
return np.clip(noisy_image, 0, 255).astype(np.uint8)

# 함수를 사용해 image에 가우시안 노이즈를 추가한 후, 그 결과를 변수에 저
gaussian_noise_img = add_gaussian_noise(image)
cv2.imshow('Original Image', image) 
cv2.imshow('Gaussian Noise', gaussian_noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 영상 비네팅 효과 Vignetting Effect
```
비네팅 효과는 이미지의 가장자리 부분이 점점 어두워지는 시각적 효과
카메라 렌즈 또는 디지털 후처리를 통해 발생할 수 있음
- 이미지의 중심부는 밝게 유지되며, 가장자리로 갈수록 어두워지는 경향
- 중심부에서의 strength 값 (강도)에 따라 이 어두워지는 정도가 결정
- 비네팅 효과는 사진이나 영상에서 주제에 집중감을 부여하고, 시선을 중심부로 끌기 위해 용
```
![image](https://github.com/user-attachments/assets/fd0e3503-99fd-492b-be3f-e707443e371b)

### 비네팅 효과 함수 코드
```
# 1. 함수 정의 및 입력 변수
# 비네팅 효과 강도를 결정하는 인자 strength 
def add_vignetting_effect(img, strength=200):

# 2. 이미지 크기 가져오기
# 이미지의 세로 길이와 가로 길이 가져옴
=> 이미지의 크기 정보를 활용해 비네팅 마스크 적용하는데 사용
rows, cols = img.shape[:2]

# 3. 중심점과 거리에 따라 가중치 적용
# 이미지의 행에 대한 좌표 배열, 이미지의 열에 대한 좌표 배열 => 격자배열 
X_result, Y_result = np.ogrid[:rows, :cols]
center_x, center_y = cols / 2, rows / 2 # 이미지의 중심점 계산
# 각 픽셀이 중심점으로부터 거리를 계산을 계산하여 distance_from_center에 저장
distance_from_center = np.sqrt((X_result - center_y)**2 + (Y_result - center_x)**2)

# 4. 최대 거리 계산
# 이미지의 중심으로부터 가장 먼 거리 계산 => 이미지의 대각선 길이 사용해 구함
max_distance = np.sqrt(center_x**2 + center_y**2)

# 5. 비네팅 마스크 생성
# distance_from_center: 이미지의 각 픽셀이 중심에서 얼마나 떨어져있는지 나타내는 거리
# max_distance: 이미지 중심에서 가장 먼 거리=이미지 대각선 길이
# 각 픽셀의 거리 값을 비율로 만듦 -> 이미지 중심부는 거리가 짧아 0, 가장가리는 멀어서 1에 가까움
# 1에서 빼면 중심부를 1에 가깝고, 가장자리는 0에 가까운 비네팅 마스크 생성
vignetting_mask = 1 - (distance_from_center / max_distance)

# 6. 비네팅 효과 강도 조절
# 가중치 적용 후 비네팅 효과 강화
# strength에 따라 비네팅 효과의 강도조절
# 기본적으로 255로 나누어 범위를 맞춘 뒤 strength에 비례에 마스크 강도조절
vignetting_mask = vignetting_mask * (strength / 255)
# 비네팅 마스크 값이 0과 1사이 있도록 제한
# 배열의 값을 지정된 범위로 자름으로써 계산된 값이 범위 해당 안 될 경우 0또는 1로 변
vignetting_mask = np.clip(vignetting_mask, 0, 1)

# 7. 각 채널에 비네팅 효과 적용
result = np.zeros_like(img) # 결과 이미지를 원본이미지와 동일한 크기, 모든 값 0인 배열로 초기
for i in range(3): # B, G, R 채널에 동일한 마스크 적용
result[:, :, i] = img[:, :, i] * vignetting_mask
return result.astype(np.uint8) # 결과 이미지를 np.uint8 타입으로 변환

# 8. 비네팅 효과 함수 호출
vignetted_image = add_vignetting_effect(image)
```
### 감마 보정 Gamma Correction
```
감마 보정은 이미지의 밝기를 비선형적으로 조절하는 방법
픽셀 값마다 다른 비율로 밝기를 조절하여, 이미지의 전체 밝기를 자연스럽게 변화
감마 보정의 공식은 출력 픽셀값 = 입력 픽셀값 ** 감마
- 감마 < 1 => 이미지가 밝아짐
- 감마 = 1 => 원본 이미지와 동일
- 감마 > 1 => 이미지가 어두워짐
- 선형적 밝기 조정(모든 픽셀을 같은 비율로 밝게 하는 것)은 자연스러움이 부족함
=> 감마 보정은 비선형적으로 각 픽셀 값을 변화시켜 자연스러운 밝기 조정 가능해짐
ex) 이미지의 어두운 부분은 덜 밝아지고, 밝은 부분은 더 밝아지며 균형있는 밝기 조정
```
![image](https://github.com/user-attachments/assets/762a53cb-d2fd-4466-b56e-11bfcc547b4c)

### 감마 보정 적용 함수 
```
import cv2
import numpy as np

g = float(input("감마 값: "))
img = cv2.imread("soccer.jpg")

# 감마 변환 수행
out = img.copy() # 원본 이미지를 복사해 out 변수에 저장 
out = out.astype(np.float32)  # 이미지의 데이터 타입을 float32로 변환
# 이미지의 픽셀 값을 0~1 사이의 값으로 변환한 후 감마 보정 적용
# 감마 값 만큼 지수승, 다시 255 곱해 원래 픽셀 범위로 변
out = (out / 255) ** g * 255  # 감마 변환 공식 적용
out = out.astype(np.uint8)  # 다시 uint8 타입으로 변환해 opencv에서 이미지로 처리

cv2.imshow("original", img)
cv2.imshow("gamma", out)
cv2.waitKey(0)
```
#### 이미지에서 255 나누면 0~1 사이로 변환
```
이미지의 픽셀 값을 255로 나누면 0~1 사이로 변환(이미지가 0에서 255사이의 값을 가짐)
=> 0은 검은색 255는 흰색 그 사이는 회색조
255로 나누는 이유는 픽셀 값을 0에서 1 사이로 정규화하기 위해서!
즉 픽셀 값 255로 나누면 원래의 픽셀 값이 소수로 변환되어 0과 1사이의 값이 
```
### 히스토그램 정보를 활용해, 사용되지 않는 픽셀 정보 강화
```
이미지는 히스토그램 정보를 사용해 이미지의 픽셀 값을 조정하고, 사용되지 않는 픽셀 정보를 강화하는 과정
- 히스토그램: 이미지의 밝기 분포를 나타내는 그래프, x축은 픽셀 값, y축운 해당 픽셀 값의 빈도
- 문제점: 이미지의 히스토그램을 보면, 특정 구간 0~50의 픽셀 값이 거의 사용되지 않음
=> 이미지에서 대비가 낮거나 어두운 부분이 많이 사용된다는 의미
- 해결책: 히스토그램 평활화는 특정 값에 집중된 픽셀 분포를 좀 더 넓고 고르게 분포시키는 과정
=> 이미지의 대비를 개선하고, 사용되지 않는 픽셀 값을 효과적으로 활용
```
```
1. 위쪽 이미지: 픽셀 값이 대부분 낮은 값(어두운 부분)에 집중
=> 이미지가 명암 대비가 낮고, 일부 정보는 거의 사용되지 않고 이씀
2. 아래쪽 이미지: 히스토그램 평활화가 적용된 결과로, 픽셀 값의 분포가 넓어지고 고르게 분포
=> 이미지의 대비가 개선, 사용되지 않던 픽셀 값들이 더 잘 활용, 이미지가 더 선명하고 확
```
![image](https://github.com/user-attachments/assets/6ef76a0f-2aa5-455f-ba38-b4d85b78fc19)

### 히스토그램 평활화 Histogram Equalization
```
1. 히스토그램
- 이미지에서 각 픽셀 밝기 값이 얼마나 많이 사용되었는지 보여주는 그래프
- 특정 구간에 픽셀 값이 몰려있을 경우, 그 부분의 대비가 낮아져 이미지가 흐릿하거나 디테일 부족
2. 히스토그램 평활화 Histogram Equalization
- 히스토그램 평활화는 이미지의 밝기 값을 더 균일하게 분포시키는 기법
- 이 과정을 통해 픽셀 값의 범위를 넓히고, 대비를 강화
=> 어두운 부분은 밝게, 밝은 부분은 더 선명하게 표현
3. 작동 원리
- 히스토그램 평활화는 밝기 채널에만 적용
(색상 정보는 유지하면서 밝기만 조정하여 이미지의 컬러 정보가 왜곡되지 않도록 하기 위함)
- RGB 컬러 이미지의 경우, LAB 또는 HSV 색공간 변환한 후, 밝기 채널에만 평활화 적용
마지막으로 다시 RGB로 변환하여 결과를 얻음 
```
```
1. 왼쪽 이미지
왼본 이미지는 대비가 낮고 흐릿하게 보임, 픽셀 값이 좁은 범위에 분포되어 있어 이미지가 흐릿함
2. 오른쪽 이미지
평활화가 적용되면 이미지의 밝기 값이 더 넓은 범위에 분포, 대비가 개선, 디테일 잘 보임
원본 이미지에 비해 밝은 부분은 더 밝고, 어두운 부분은 더 뚜렷해지는 효과 
```
![image](https://github.com/user-attachments/assets/57688a1f-ed7a-4bff-98af-fc5f05da7dd3)

### 히스토그램 평활화 코드 >> 
```
import cv2
# 컬러 이미지 읽기
image = cv2.imread('hazy.jpg') # 이미지 BGR 형식 저장, 기본 색공간 
# BGR에서 LAB 색 공간으로 변환
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # LAB 색공간 변환 
# L, A, B 채널 분리
L, A, B = cv2.split(lab_image)

# L 채널에 히스토그램 평활화 적용
# cv2.equalizeHist() 이미지의 히스토그램 평활화를 적용하는 함수
# 히스토그램 평활화는 픽셀의 밝기 값 분포를 평탄화하여 대비를 개선하는 기법
# L채널에만 적용하므로, 이미지의 색상 정보는 유지되고 밝기 정보만 개선  
L_eq = cv2.equalizeHist(L)

# 평활화된 L 채널과 원본 A, B 채널을 다시 합침
# 원래의 채널 두 개와 L_eq 채널 합쳐 LAB 이미지 만듦
lab_eq_image = cv2.merge([L_eq, A, B])

# LAB에서 다시 BGR로 변환
equalized_image = cv2.cvtColor(lab_eq_image, cv2.COLOR_LAB2BGR)
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
```
### 콘트라스트 스트레칭 Contrast Stretching
```
개념 >>
- 콘트라스트 스트레칭은 이미지의 최소값과 최대값을 기준으로 픽셀 값을 선형적으로 변환해, 전체적인 대비 증가
- 이미지의 어두운 부분은 더 어둡게, 밝은 부분은 더 밝게 만들어 전체 이미지의 대비가 향상

```
#### 새로운 픽셀 값의 공식
#### New pixel = (pixel_value-min_val) / (max_val-min_val) * 255
```
- pixel_value: 현재 픽셀 값
- min_val: 이미지에 사용된 최소 픽셀 값
- max_val: 이미지에 사용된 최대 픽셀 값
=> 이미지의 픽셀 값을 0에서 255 사이로 재분포하는 과정
```
```
1. 왼쪽 이미지 = 원본
- 원본 이미지는 전체적으로 밝기 범위가 좁고, 대비가 낮아 흐릿함
이미지가 어둡고 명암비가 부족해보이는 전형적인 경우

2. 오른쪽 이미지
- 콘트라스트 스테리칭을 적용한 후, 이미지의 밝기 범위가 전체적으로 확장되면서,
어두운 부분은 더 어둡게, 밝은 부분은 더 밝게 됨
- 이미지의 대비가 증가하고, 전체적인 선명도와 명암비 개선 
```
![image](https://github.com/user-attachments/assets/9180df08-e927-40b9-9109-9771a1f8bd99)

### 콘트라스트 스트레칭 함수 부분 >> 
```
import cv2
import numpy as np
# 이미지 읽기
image = cv2.imread('hazy.jpg')
# 이미지의 최소값과 최대값 찾기
min_val = np.min(image) # 이미지 배열에서 최소값 = 가장 어두운 부분
max_val = np.max(image) # 이미지 배열에서 최대값 = 가장 밝은 부분

# 콘트라스트 스트레칭 적용 (새로운 범위: 0~255)
stretched_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
cv2.imshow('Original Image', image)
cv2.imshow('Contrast Stretched Image', stretched_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
```
## 4. 영상 변환
### 크기 조절 Scaling과 보간법 Interpolation
```
이미지의 크기를 조절할 때, 새로운 픽셀 값을 어떻게 계산할 것인가에 따라 보간법 달라짐
보간법은 기존의 픽셀 값을 기반으로 새로운 픽셀 값을 계산하여, 크기를 조정한 이미지의 품질 결정
```
```
이미지 세 개 비교 >>
1. 최근접 이웃: 확대된 이미지에서 픽셀이 도드라져 보이며, 경계가 뚜렷(계단 현상)
2. 양선형 보간: 확대된 이미지가 상대적으로 부드럽고 자연스러우며, 경계 부드러움
3. 양3차보간: 가장 자연스러운 확대 -> 픽셀 간의 경계가 가장 부드럽고 이미지 품질이 유
```
![image](https://github.com/user-attachments/assets/a012726c-38d6-480c-bbef-5e00a5bfee0d)
### 보간법의 종류 >>
```
1. 1D Nearest-Neighbor (1차원 최근접 이웃)
: 새로운 픽셀 위치에서 가장 가까운 픽셀 값을 그대로 복사하여 사용하는 방법
계산이 매우 빠르고 간단하지만, 확대된 이미지에서 계단 현상과 같은 불연속적인 픽셀 경계 나타냄
2. Linear Interpolation (선형 보간)
: 두 점 사이에 있는 새로운 픽셀 값을 선형적으로 계산
1D에서만 적용되는 방식, 근처 픽세 값을 기반으로 직선을 그려 중간값 보간해 부드러움 추가
3. Cubinc Interpolation (3차원 함수 보간)
: 3차 함수를 사용해 두 점 사이의 값 보간
직선보다 부드러운 곡선으로 보간, 선형 보간보다 더 자연스럽게 픽셀 값 계산
4. 2D Nearest-Neighbor (2차원 최근접 이웃)
: 1차원 최근접 이웃의 2차원 버전으로, 가장 가까운 픽셀 값을 사용하는 방식
가장 가까운 픽셀 값을 그대로 복사하기 때문에 이미지 확대 시 경계선이 뚜렷 (계단 현상)
5. Bilinear Interpolation (양선형 보간)
: 주변 4개의 픽셀 값을 사용해 새로운 픽셀 값 계산
2D 이미지에서 널리 사용되며, 주변 픽셀 값 선형으로 평균내어 부드럽게 보간
6. Bicubic Interpolation (양3차보간)
: 주변 16개의 픽셀 값 사용하여 새로운 픽셀 값 보간하는 방식
2D이미지에서 가장 자연스러운 보간법 중 하나, 주위 픽셀 값을 3차함수로 계산해 매우 부드러운 이미지  
```
#### 요약 >>>>
```
- 1D 및 2D Nearest-Neighbor: 계산이 빠르지만 이미지 품질이 낮으며, 계단 현상 발생
- Linear: 간단한 선형 보간법으로, 두 점 사이의 값을 직선을 연결해 계싼
- Cubic: 곡선을 기반으로 더 부드러운 결과 제공
- Bilinear: 주변 4개의 픽셀을 고려하여 새로운 픽셀 값을 계산하므로 부드러운 보간
- Bicubic: 주변 16개의 픽셀을 사용해 가장 자연스럽고 부드러운 보간 제공, 이미지 확대에 유
```
![image](https://github.com/user-attachments/assets/25751c88-7507-4768-8105-cab1e42b63e8)

### 보간법 3가지 주요 방식 >>
```
1. 최근접 이웃 보간법 Nearest Neighbor Interpolation
: 새로 생성된 픽셀의 값이 가장 가까운 픽셀 값을 그대로 복사하는 방법
방식 >>
이미지 확대 시, 새로 생긴 빈 픽셀을 인접한 기존 픽셀로 채우는 방식
Ex) 1과 2사이에 새로운 픽셀 값 넣을 때 1 또는 2 중 하나로 유지

2. 양선형 보간법 Bilinear Interpolation
: 새로운 픽셀 값이 주변의 4개 픽셀 값을 선형적으로 보간하여 계산
방식 >>
그래프에서 X좌표의 중간 위치에 새로운 픽셀을 추가할 때 주변 두 값 (a,b) 사이의
선형적 관계를 통해 중간 값 f(x) 계산

3. 양3차 보간법 Bicubic Interpolation
: 주변 16개의 픽셀의 값을 사용해 3차 함수 곡선을 그려 새로운 픽셀 값을 계산하는 방법
방식 >>
3차함수 곡선 f(x)를 사용해 픽셀 사이 중간 값 계산, 주변 픽셀을 이용해 곡선의 형태 만들어 얻음 
```
### 보간법 적용 함수 >>
```
import cv2
import numpy as np
image = cv2.imread('soccer.jpg')

# 이미지 자르기 crop: 배열 슬라이싱 방식 
crop_image = image[150:200, 150:200]
resize_dim = (200, 200)

# Nearest Neighbor 방식으로 리사이즈
# 계산이 빠르지만 이미지 품질 낮고, 계단 현상 발생 가능성
resize_nn = cv2.resize(crop_image, resize_dim, interpolation=cv2.INTER_NEAREST)

# Bilinear 방식으로 리사이즈
# 주변 4개의 픽셀 값을 선형적으로 보간하여 부드럽게 크기 조정하는 방식
=> 이미지의 경계선이 덜 도드라지고 부드럽게 나타남 
resize_bilinear = cv2.resize(crop_image, resize_dim, interpolation=cv2.INTER_LINEAR)

# Bicubic 방식으로 리사이즈
# 주변 16개의 픽셀 값을 3차함수로 보간하여 더 부드럽고 자연스러운 결과 제공
=> 계산은 복잡하지만, 이미지 확대 시 품질이 가장 우수 
resize_bicubic = cv2.resize(crop_image, resize_dim, interpolation=cv2.INTER_CUBIC)

result = cv2.hconcat([resize_nn, resize_bilinear, resize_bicubic])
cv2.imshow('Resized Comparison', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 여러가지 기하 변환
```
1. 회전 (Rotation): 이미지의 기준점을 중심으로 특정 각도만큼 회전시키는 변환

2. 이동: 이미지를 좌우, 상하로 이동시키는 변환

3. 크기 변경: 이미지를 확대하거나 축소하는 변환

4. 이동과 크기 변경 (Translation + Scaling): 이미지 이동하고 크기 변경하는 복합 변환
```
![image](https://github.com/user-attachments/assets/344f660d-bfac-448f-ba6b-88f99a358bad)
#### 이동, 회전, 크기 등 동차행렬 연산을 통해 변환
```
- 이동 변환: 객체를 x와 y방향으로 각각 t_x, t_y만큼 이동시키는 변환
- 회전 변환: 원점을 기준으로 세타만큼 회전시키는 변환, 반시계 방향각도
- 크기 조절 변환: 객체의 크기를 x축과 y축에 대해 s_x, s_y배 확대하거나 축소하는 변
```
![image](https://github.com/user-attachments/assets/ad85abbd-d8e4-4d77-88fe-01afb62a99a9)
#### 이미지 이동 코드 >>
```
import cv2
import numpy as np

img = cv2.imread('soccer.jpg')
h, w, c = img.shape # 이미지의 높이, 너비, 채널 정보 반환 

# translation matrix 2*3 행렬 정의
# x축으로 -100픽셀, y축으로 -200픽셀을 이동
# np.float32()는 변환 행렬을 32비트 부동 소수점 형식으로 만듦
M = np.float32([[1, 0, -100], [0, 1, -200]])

# affine 변환 적용
# cv2.warpAffine(): 변환 행렬 M을 사용하여 원본 이미지에 img에 affine 변환 적용 함수
dst = cv2.warpAffine(img, M, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Translation', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### 이미지 이동 Translation
```
- 이미지 이동은 이미지의 각 픽셀을 x축과 y축으로 특정 거리만큼 이동시키는 변환
- 이동변환은 이미지의 모양과 크기는 변하지 않지만, 좌표 위치가 변동
- 이때, 이미지를 이동하는 행렬 변환 사용
```
![image](https://github.com/user-attachments/assets/2915b393-d54d-4342-aa90-8bf4716aa993)

### 이미지 회전 및 크기 변환 코드 
```
import cv2

img = cv2.imread('soccer.jpg')
h, w, c = img.shape

# cv2.getRotationMatrix2D(): 이미지를 회전시키기 위한 2D 회전 변환 행렬 생성
# 첫 번째 인자: 회전의 중심점 지정, 이 코드에서는 이미지의 중심이 기준
# 두 번째 인자: 회전 각도 지정, 반시계로 45도 회전
# 세 번째 인자: 0.5배 축소 의미, 이미지 크기를 절반으로 줄이는 스케일링 값 설정 
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 0.5)

# cv2.warpAffine() 함수는 변환행렬 M을 사용해 원본 이미지 img에 회전 및 크기 변환을 적용하는 함수
dst = cv2.warpAffine(img, M, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Rotation', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://github.com/user-attachments/assets/5400e3f8-7336-427e-a321-d84330cd8e55)


## 5. 영상 품질 측정 
#### PSNR (최대 신호 대 잡음 비, Peak Signal Noise Ratio)
```
PSNR은 이미지나 영상의 화질 손실을 측정하기 위해 사용되는 지표
신호의 최대값에 비해 잡음(노이즈)가 얼마나 있는지를 나타내며, 두 이미지 간의 품질 차이 평가
PSNR는 데시벨 dB단위로 표현, 값이 클수록 품질이 좋다는 것을 의미
```
![image](https://github.com/user-attachments/assets/139faee0-fafe-4b6b-bfe9-604c03f4b799)
```
- PIXEL_MAX는 이미지의 최대 픽셀 값, 8비트 이미지는 0~255 가지므로 255
- MSE(Mean Squared Error) 두 이미지 간의 차이를 계산하는 지표, 각 픽셀 값의 차이 제곱 평균
=> 작을수록 두 이미지가 더 유사
=> PSNR은 MSE값이 작을수록 (이미지 간의 차이가 적을수록) 값이 커지며, 높은 PSNR값은 더 높은 화질 의미
```
```
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)  # 두 이미지 간의 MSE 계산
    if mse == 0:  # 두 이미지가 동일할 경우 MSE가 0이므로 PSNR은 무한대가 됨
        return 100  # 완벽한 품질이므로 100으로 설정
    PIXEL_MAX = 255.0  # 8비트 이미지의 최대 픽셀 값
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))  # PSNR 계산 공식
    return psnr
```
### SSIM (구조적 유사도 측정 지표, Structural Similarity Index Measure)
```
두 이미지 간의 구조적 유사도를 평가하기 위한 지표, PSNR보다 인간의 시각적 특성을
더 잘 반영하는 평가방법 SSIM은 이미지의 대비, 밝기, 구조 요소를 고려해 유사도 측정
```
![image](https://github.com/user-attachments/assets/bc92bb0e-35be-46f3-894d-ea5bd3ce115e)
```
1. 1에 가까울수록 두 이미지가 구조적으로 매우 유사함
2. 0에 가까울수록 두 이미지 간의 구조적 차이가 큼
3. SSIM과 PSNR은 다르다
PSNR은 두 이미지 간의 픽셀 값 차이에 집중하지만, SSIM은 구조적 유사성을 평가
두 지표는 항상 비례하지 않으며, PSNR이 높아도 SSIM이 낮을 수 있음
이미지의 구조적 요소가 유지되지 않았다는 것을 의미 
```
