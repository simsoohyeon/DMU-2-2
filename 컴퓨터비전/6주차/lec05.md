![image](https://github.com/user-attachments/assets/54ede6c9-2c9b-4f6c-81fe-de26cf33572c)# 1. 에지 검출 개요 
```
에지 edge 검출은 이미지에서 물체의 경계를 찾아내는 기법
이미지 분석의 기본 단계로, 컴퓨터 비전에서 중요한 역할
목적: 물체의 윤곽선을 파악하고 영역을 분리하여, 이미지 내의 개체나 구조를 쉽게 분석
활용 분야: 이미지 분석, 의료 영상 처리, 얼굴 특징 처 
```
![image](https://github.com/user-attachments/assets/095d6d3d-3ecb-4b02-8f50-af09761b1d66)
## 에지 검출 알고리즘
```
이미지에서 경계를 감지하는 기법, 물체의 형태를 구분하는데 중요
- 원리: 이미지에서 명암이 급격하게 변하는 부분을 에지로 간주
물체 내부는 일반적으로 명암차이가 적어 평탄, 물체의 경계에서는 명암이 크게 변화
- 알고리즘 동작: 픽셀 간 명암 변화를 분석해, 변화가 큰 부분을 에지로 검출
이미지의 일부분을 확대하여 보면, 각 픽셀의 명암 값이 다르며, 에지 검출은 이 값을 기반으로 경계 찾음
- 활용: 에지 검출을 통해 물체를 파악하거나, 이미지 내의 특정 개체 분리해 분
```
![image](https://github.com/user-attachments/assets/9a496066-f4e8-42f3-ac68-5eaa4fb968ad)

# 2. 미분 연산을 통한 에지 출
## 에지 검출을 위해 1차 미분 개념
```
1. 1차 미분을 통한 기울기 변화 탐지
이미지는 픽셀로 구성되어있어 연속적이지 않기 때문에, 미분 대신 차분 연산 사용해 기울기 계산
밝기 값이 급격히 변화하는 부분은 경계를 나타내며, 이러한 변화는 1차 미분으로 감지

2. 에지 검출 필터
prewitt 필터와 sobel 필터는 이미지에서 에지를 검출하는데 사용되는 대표적인 필터
이 필터들은 이미지의 특정 방향으로의 기울기를 계산해 경계를 강조
sobel 필터는 prewitt 필터보다 더 정교하게 에지를 추출하도록 개션된 형태,
특히 소음있는 이미지에서도 더 안정적인 경계 추출

3. sobel x와 sobel y 필터 필터
sobel x는 수평 방향의 변화를 강조하여 수직 에지 검출
sobel y는 수직 방향의 변화를 강조하여 수평 에지 검출
이 두 필터를 결합하면 이미지의 전체적인 경계 에지 감
```
![image](https://github.com/user-attachments/assets/b4bd6a2d-0e27-4268-b1e6-221281ad4b04)
## Sobel 필터를 이용해 이미지의 에지 검출
```
import cv2
# 이미지 불러오기
img = cv2.imread('img/box.jpg')

# Sobel 에지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)  # x방향
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)  # y방향
# sobel 필터 적용해 x축과 y축 방향의 에지 검출
# cv2.Sobel() 함수의 인자: 첫 번재 입력 이미지,
# -1은 출력 이미지의 데이터 타입을 입력 이미지와 동일하게, 세 네번째 인자 1,0은 x방향, 0,1은 y방향
# ksize=3 커널사이즈를 지정하여 필터 크기 설정

edge_strength = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# sobelx, sobely 가중 합쳐서 edge_strength 이미지 생성
# 각각 0.5의 가중치를 적용하여 x축과 y축의 경계를 균등하게 합침 

cv2.imshow('origin', img)
cv2.imshow('sobel x', sobelx) # x방향 에지만 검출된 이미지
cv2.imshow('sobel y', sobely) # y방향 에지만 검출된 이미지
cv2.imshow('edge strength', edge_strength)

cv2.waitKey(0)

결과 이미지 해석 >>
sobel x: 수평 방향으로의 밝기 변화가 두드러진 부분 강조
sobel y: 수직 방향으로의 밝기 변화가 두드러진 부분 강조
edge strength: x와 y방향 에지 결합해 전체 경계선 보여줌
```
![image](https://github.com/user-attachments/assets/1d53d2a1-c745-44ba-ba0b-45532924045e)

## 2차 미분을 통한 에지 검출
```
1. 1차 미분을 통한 기울기 변화 탐지
1차 미분은 이미지에서 밝기 값의 변화가 큰 부분 = 기울기를 측정해, 밝기 변화가 큰 부분을 경계로 식별
하지만 1차 미분만으로는 에지가 정확히 어디서 시작되고 끝나는지 불명확 (정확한 위치 불명확)
에지 근처에서 기울기가 급격히 증가하지만, 에지가 어디에서 분명하게 구분되는지 명확하지 않음

2. 2차 미분 (기울기 변화의 중심 잡음 -> 에지의 정확한 위치 결)
기울기의 변화량을 측정
1차 미분의 값이 증가하거나 감소하는 지점에서 2차 미분은 양에서 음, 음에서 양으로 바뀌는 특성
이를 통해 에지의 중심을 더 명확하게 검출함
2차 미분의 결과는 에지가 시작되거나 끝나는 정확한 지점을 나타내는 강한 신호(0을 지나는 교차점)
에지의 위치를 정확하게 결정하는데 유용

3. 필터 연산
[-1,1]와 같은 1차 미분 필터를 사용해 기울기 측정, 다시 한 번 [-1,1] 적용해 2차 미분 구함
원래 이미지에서 1차 미분을 적용한 결과로 두 개의 에지 구간이 나타나지만,
2차 미분을 통해 에지가 발생하는 정확한 위치가 명확
```

![image](https://github.com/user-attachments/assets/39a8e030-7a9a-419d-83d1-9051f64af5dc)

## 2차 미분을 이용한 에지검출의 예시, 라플라시안 Laplacian 연산
```
1. 라플라시안 연산
라플라시안연산은 이미지의 기울기 변화를 2차 미분하여 에지 검출
1차 미분을 한 번 더 미분하여, 변화가 있는 곳의 중심을 더 명확하게 검출
이 연산은 특힝 에지의 중심 잘 검출하므로, 에지 검출을 더 정교하게 하는데 유용

2. open cv의 cv2.Laplacian 함수
cv2.Laplacian 함수는 입력 이미지에 라플라시안 필터 적용하여 2차 미분 수행
함수의 두 번째 인자는 ddepth 출력 이미지의 데이터 타입 설정, cv2.CV_8U는 8비트 unsigned int 형식
```
```
laplacian = cv2.Laplacian(img, cv2.CV_8U)
# cv2.CV_8U는 출력 이미지의 정밀도를 8비트 unsigned int 형식으로 설정

결과 이미지 해석 >>
Laplacian 결과: 상자 경계 부분이 뚜렷하게 강조, 상자의 윤곽선이 정밀하게 나타나고 불필요한 부분 배 
```
![image](https://github.com/user-attachments/assets/5aca9d9f-7174-4252-b45e-f3f020e70c80)

# 3. 캐니 에지 검출
```
- 에지를 검출하면서 잡음을 줄이는 기법, 에지의 정확성을 높이고 불필요한 잡음 제거, 명확한 에지 표현

1. 목표
낮은 오류울: 실제 에지인 부분만을 정확히 검출, 불필요한 잡음이나 잘못된 에지 검출 최소화
에지 위치 정확도: 에지의 위치를 정확하게 검출해 한 점으로 표현
단일 에지 표현: 불필요한 중복 에지 대신, 하나의 점으로 표현할 수 있는 명확한 에지 추출

2. 에지 검출 과정
가우시안 스무딩: 에지 검출 전에 이미지를 부드럽게 만들어, 잡음이 에지 검출에 영향을 주지 않도록 함
그래디언트 계산: 이미지에서 기울기를 계산하여 에지의 방향과 세기를 파악, sobel 필터 적용
비최대 억제: 에지의 강도가 가장 큰 부분만 남기고, 나머지 부분 억제 -> 에지의 위치를 더욱 선명하게 만듦
NMS, Non maximum suppression
이중 임계값 사용: 두 개의 임계깞을 설정하여 강한 에지와 약한 에지 구분
강한 에지는 유지하고, 약한 에지는 강한 에지와 연결되어 있을 때만 유지하여 불필요한 잡음 줄임

3. 비최대 억제
비최대 억제는 특정 위치에서 에지의 세기가 주변보다 크지 않으면 억제하는 방식
에지 방향을 따라 최대값이 아닌 화소들은 에지에서 제외되어 단일하고 명확한 에지만 남
```
![image](https://github.com/user-attachments/assets/ac05935b-1722-4634-bc98-44d41b3ca36e)
## 캐니 에지 검출을 적용한 결과 
```
canny = cv2.Canny(img, 100, 150)
# 두 번재와 세 번째 인자
# 100은 낮은 임계값, 이 값보다 낮은 그래디언트=기울기는 에지로 간주되지 않음
# 150은 높은 임계값, 이 값보다 높은 그래디언트는 확실한 에지로 간주

결과 이미지 >> 캐니 에지 검출을 적용한 결과
이미지의 윤곽이 흰색 선, 배경은 검은색으로 표현된 이진 이미지
박스의 경계선이 뚜렷하게 검출되어 상자 형태가 명확히 드러남
잡음 없이 단순화된 에지 정보만을 남겨 이미지의 구조를 효과적으로 표현
```
![image](https://github.com/user-attachments/assets/4533e775-d294-4f37-962a-0566d31e64ed)

# 4. 직선 검출
## 허프 변환 hough transform을 이용한 직선 검출
```
1. 허프 변환의 목적
에지 검출의 결과에서 직선의 형태를 찾아내기 위해 사용
이미지의 여러 점들을 분석하여, 이 점들이 특정한 직선에 많이 모여있는 경우 해당 직선을 검출

2. 원리
이미지 상의 모든 에지 점에 대해 직선 방정식을 사용해 직선 후보 생성
각 점에서 직선의 가능성을 확인하여, 점들이 특정한 직선이 많이 교차하면 그 직선을 검출
이때 직선의 표현을 위해 두 가지 값 사용 >>
rho 원점에서 직선까지의 최소 거리 / theta 직선이 원점과 이루는 각도
이 두 값을 기반으로 직선의 후보를 설정, 특정 점들이 해당 직선에 많이 포함되면 최종 직선으로 판단

3. 직선 검출 과정
에지 이미지에서 각 에지 점에 대해 다양한 직선 후보 생성
이 점들이 모여 특정한 직선을 이루는 경우, 그 직선을 검출해 결과에 포
```
![image](https://github.com/user-attachments/assets/a951ea03-f758-4c6c-8b33-0fefd01928dc)
## 허프 변환을 사용해 직선 검출
```
1. 직선 검출을 위한 에지 기반 접근
에지 검출을 통해 이미지에서 윤곽선을 추출한 후, 이러한 에지들이 직선 형태를 이루는지 확인
각 에지점이 특정 직선에 얼마나 일치하는지 계산하여 직선을 추출

2. 허프 변환의 원리
에지 이미지의 각 점에 대해 직선 후보 설정, 이 후보들 중에서 많이 교차하는 직선을 최종적으로 선택
rho와 theta라는 두 가지 값으로 직선을 표현
rho 직선과 원점 사이의 최소 거리, 직선이 원점에서 얼마나 떨어져 있는지 나타냄
theta는 직선이 원점과 이루는 각도, 직선의 기울기 방향

3. 에지 점들이 같은 직선에 있을 때
여러 에지 점들이 rho와 theta가 동일한 직선 후부에 많이 겹친다면, 해당 직선을 최종 직선으로 판단
파란색 선처럼 여러 에지 점들이 동일한 직선 후보에 일치하면, 그 직선이 검출 
```
![image](https://github.com/user-attachments/assets/c1fc3a9c-b7d2-4550-8d21-e7594182ab7d)
## 에지 방향과 직선 검출의 차이
```
1. 에지 방향에 따른 차이점
에지가 기울어지거나 방향이 다를 때, 각 에지점은 허프공간에서 다른 rho와 theta값을 가짐
허프 변환은 모든 에지 점의 rho와 theta 값을 확인하면서 직선 후보 만듦
만약 에지 방향이 서로 다르다면, 동일한 직선으로 인식되기 어려움
아래의 파란색처럼 기울어진 에지의 경우, 원점과 이루는 각도가 달라져 다른 직선으로 인식

2. rho와 theta 값의 일관성
동일한 직선에 속하려면 여러 에지 점들이 rho와 theta값을 가져야 함
에지가 여러 방향으로 흩어져 있다면, rho와 theta값이 상이해 직선으로 검출되지 않거나 여러 개의 직선

3. 허프 변환의 특징
허프 변환은 이미지 내에서 직선 형태가 얼마나 일관되게 이어지는지에 따라 직선을 인식
에지의 방향이 균일하고 일관된 경우, 하나의 직선으로 인식되기 쉽지만 그렇지 않으면 다른 직선으로 검출될 가능성
```
![image](https://github.com/user-attachments/assets/cf8389f4-f960-47d0-bfa4-d44016686eaf)
## 허프 변환 사용한 코드
```
lines = cv2.HoughLines(canny, 1, np.pi/180, 80)
# canny는 입력 이미지, 에지 검출된 이진화 이미지
# 1은 rho의 해상도로, 직선의거리 간격을 1픽셀 단위로 설정
# np.pi/180: theta의 해상도로 각도는 1도 단위로 설정
# 80: 임계값으로 이 값 이상으로 겹치는 직선 후보만 직선으로 인식
=> lines 리스트에 검출된 직선의 rho와 theta 값 저장  

# 검출된 직선 그리기
for line in lines:
    rho, theta = line[0] # 직선의 시작점과 끝점 계산
    a = np.cos(theta) # 방향 계산
    b = np.sin(theta) # 방향 계산
    x0 = a * rho # 직선의 중심점 나타내는 좌표
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b)) 
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.line() 통해 (x1,y1)~(x2,y2)까지의 직선을 img 이미지 위에 초록색으로 그리고 선두께 정

```
![image](https://github.com/user-attachments/assets/a1dbab27-b095-4bce-8075-a5feb6364674)

# 5. 컨투어 검출 
## findContours 함수 사용해 이미지의 경계선 검출
```
이미지에서 물체의 외곽선 찾고, 이를 이용해 물체의 윤곽을 그리기 위해 사용
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 이미지 그레이스케일 변환, 컨투어 변환은 일반적으로 흑백 이미지에서 이루어짐

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# 이진화 처리: 그레이스케일 이미지 gray를 이진화하여 binary 변수에 저장
# 127을 임계깞으로 설정, 127보다 크면 흰색(255), 작으면 0(검은색)으로 변환해 이진화 이미지 생성
# cv2.THRESH_BINARY은 단순한 이진화 방법을 적용한다는 의미

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 컨투어(윤곽선 검출) cv2.findContours 함수는 이진화된 binary이미지에서 외곽선 찾아 contours에 저장
# cv2.RETR_TREE 모든 계층의 컨투어를 찾음
# cv2.CHAIN_APPROX_SIMPLE 윤곽선을 단순화하여 꼭 필요한 점들만 저장
# 이 함수는 모든 윤곽선의 목록을 contours에, 계층 구조 정보를 hierarchy에 저장

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
# cv2.drawContours함수는 img위에 윤곽선 contours그림
# -1은 모든 윤곽선 그린다는 의미, (0,255,0)은 초록색으로 지정, 윤곽선 두께 3

결과 이미지 >>
binary 그레이스케일로 변환된 후 이진화된 이미지, 물체와 배경이 명확히 구분되어 윤곽선 검출 용이
contour 접시와 음식의 경계가 강조되어 검출된 윤곽선 잘 드러
```
![image](https://github.com/user-attachments/assets/8d00d448-5457-4266-8e5a-61cd10bf79af)


# 6. 이미지 분할 개요
```
1. Semantic Segmentation 의미론적 분할
각 카테고리마다 서로 다른 색으로 픽셀을 구분
예를 들어, 사람은 모두 한 색, 배경은 다른 색으로 표시

2. Instance Segmentation 인스턴스 분할
같은 카테고리 내에서도 개별 객체=인스턴스를 서로 다른 색으로 구분
에를 들어, 여러 사람이 있을 때 각 사람을 다른 색으로 표시

3. Panoptic Segmentation 파노픽 분할
카테고리 별로 분할하면서, 같은 카테고리 내 개체도 서로 다른 색으로 구분
Semantic과 Instance Segmentation을 결합한 방식, 객체와 배경을 모두 세밀하게 구분분
```
![image](https://github.com/user-attachments/assets/f9d0dfb7-05cc-4211-927d-2338eec7d446)


# 7. 이진화 기반 영역 분할 
```
1. 이진화 Binary Thresholding
이진화는 그레이스케일 이미지의 픽셀 값이 특정 임계값을 기준으로 두 그룹으로 나뉘는 과정
이 과정에서 임계값보다 큰 값은 최대값으로 설정, 임계값보다 작은 값은 0으로 설정
이로 인해 검은색과 흰색으로 구분되는 이진화 이미지 생성

2. cv2.threshold 함수
src 입력 이미지, 반드시 그레이스케일 이미지
thresh 임계값, 이 값을 기준으로 픽셀이 나뉨
maxval 임계값 넘는 픽셀에 할당할 최대값, 일반적으로 흰색
type: 이진화 방식 선택
  cv2.THRESH_BINARY
  cv2.THRESH_BINARY_INV
```
![image](https://github.com/user-attachments/assets/6bf4466c-3680-4605-8c93-afcd24c75508)

# 8. 연결요소와 모폴로지 연산
```
1. 연결요소 분석
이진화된 이미지에서 흰색이나 검은색으로 연결된 픽셀 그룹을 찾아 각각을 독립된 객체로 인식하는 과정
이러한 연결된 픽셀 그룹은 이미지에서 객체나 특정 영역을 나타냄

2. 연결성 종료
4-연결성: 각 픽셀을 중심으로 상하좌우 네 방향에 연결된 픽셀 확인, 대각선 방향으로 연결되지 않은 것으로 간주
8-연결성: 대각선 방향까지 포함하여 8방향으로 연결된 픽셀을 모두 확인
이 방식은 더 넓은 범위의 연결을 확인하는데 사용
```
![image](https://github.com/user-attachments/assets/c97380dc-818f-4e0f-b55f-091457c532c5)

## cv2.connectedComponentsWithStats 함수
```
이 함수는 이진화된 이미지에서 연결된 객체를 찾아내고, 각 객체의 정보 반환
입력 이미지는 binary_image는 이진이미지여야 하며, 검출된 객체 분석

파라미터 설명 >>
connectivity: 연결성을 지정하는 값 4또는 8
4 연결성: 상하좌우로 연결된 픽셀을 하나의 객체로 간주
8 연결성: 대각선 방향까지 8방향으로 연결된 픽셀을 하나의 객체로 간주

반환값 설명 >>
cnt 검출된 객체의 개수, labels 각 픽셀에 대해 객체 번호가 부여된 레이블 이밎
stats 각 객체의 통계 정보를 포함한 배열, 객체의 바운딩 박스 정보, 픽셀 개수 등이 포함
centriods 각 객체의 중심 좌표

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
```
## 이진화된 이미지에서 연결된 객체를 찾고, 각 객체의 바운딩 박스를 이미지에
```
import cv2

img = cv2.imread('lena.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

_, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# 200을 임계값으로 사용해 200 픽셀 이상인 것은 흰색으로, 미만은 0, 검은색으로 변환

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
# cv2.connectedComponentWithStats 함수를 사용해 연결된 객체 검출
# connectivity=8 8방향 연결을 기준으로 객체를 인식, cnt 검출된 객체의 수
# labels 각 픽셀에 객체 번호가 할당된 레이블 이미지, stats 각 객체의 통계 정보, 바운딩 박스 좌표와 크기 등
# centroids 각 객체의 중심 좌표표

for i in range(1, cnt): # 객체 수 cnt만큼 반복하여, 첫 번째 객체(배경) 제외하고 각 객체의 바운딩 박스 그림
    (x, y, w, h, area) = stats[i] # stat[i]은 객체의 바운딩 박스 정도로, (x,y)는 좌상단 좌표
    # w와 h는 폭과 높이, area는 객체의 면적
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.rectangle 함수를 사용해 객체의 바운딩 박스를 초록색으로 그리고, 두께는 2로

cv2.imshow('Binary', binary)
cv2.imshow('Connected Components', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

결과 해석 >>
bianary 창: 이진화된 이미지 표시, 현재 픽셀이 객체로 인식
connected components 창: 각 객체의 바운딩 박스가 초록색 사각형으로 표시된 이미지 
```
![image](https://github.com/user-attachments/assets/6814b84c-a610-4634-b21d-ceec911e6685)

## 모폴로지 연산
```
이미지 처리에서 구조적 요소를 사용해 객체의 모양이나 구조를 조작하는 기법
주로 이진화된 이미지에서 특정한 형태학적 변환을 적용하여 객체의 경계나 노이즈를 조정할 때 사용

1. 구조 요소
모폴로지 연산에서 객체의 형태를 변경하기 위해 사용하는 커널 구조
사각형, 십자형, 타원형 등 다앙햔 형태가 있으며, 각 구조 요소의 형태에 따라 연산 결과 달라짐

2. 주로 모폴로지 연산 종류
- 팽창 Dilation
객체의 크기 확장
작은 구멍을 메우고, 객체의 경계선이 확장
이미지 내에서 밝은 영역이 확대

- 침식 Erosion
객체의 크기를 줄임
작은 노이즈가 제거되고, 객체의 경계가 깎임
이미지 내에서 밝은 영역이 줄어들며, 작은 점 형태의 노이즈 제거

- 열림 Opening
침식 후 팽창을 적용
작은 노이즈가 제거되며, 객체의 원래 크기는 유지
노이즈 제거에 유용, 얇은 연결부가 끊어질 수 있음

- 닫힘 Closing
팽창 후 침식을 적용
작은 구멍이 메워지며, 객체의 경계선이 부드럽게 처리
객체의 경계선이 매끄럽게 하고, 작은 구멍을 메우는데 사
```

![image](https://github.com/user-attachments/assets/9354fbf6-29e4-41e0-919e-3b13b36ed6f1)

## 1. 팽창 Dilate 연산 
```
기능: 객체의 경계선을 다라 픽셀을 추가하여 객체를 확장하는 연산
작동 방식:
구조 요소(커널) 안에 하나라도 1이 포함되어 있으면 중앙 픽셀을 1로 설정
객체의 경계가 확장되어 작은 구멍이나 틈이 메워지고, 밝은 영역이 커짐

dilated = cv2.dilate(binary_image, kernel, iterations=1)
# kernel 구조요소, iterations=1 연산을 한 번 반복, 횟수를 늘리면 더 많이 확장
```
## 2. 침식 Erode 연산
```
기능: 객체의 경계선을 따라 픽셀을 제거해 객체를 축소하는 연산
작동 방식:
구조 요소 안의 모든 값이 1일 경우에만 중앙 픽셀로 1로 유지,
하나라도 0이 있으면 중앙 픽셀 0으로 설정 -> 객체 축소
객체의 경계가 줄어들어 작은 노이즈가 제거되고, 밝은 영역이 줄어듦

eroded = cv2.erode(binary_image, kernel, iterations=1)
# kernel 구조 요소, iterations은 연산을 한 번 반복 
```
### 팽창: 객체를 확장해 구멍이나 틈을 메우는데 유용
## 침식: 객체를 축소하여 작은 노이즈 제거하는데 유
![image](https://github.com/user-attachments/assets/8b2c5e1a-01a4-4915-ab84-87496a1c80dc)
## 3. 열림 Opening 연산
```
기능: 침식 후 팽창을 수행하는 연산
용도: 작은 노이즈 제거에 효과적
객체의 모양을 유지하면서 이미지에서 작은 불필요한 점 형태 노이즈를 없앨 수 있음

opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
# cv2.MORPH_OPEN은 열림 연산을 지정하는 플래스, kernel 연산에 사용할 구조 요소 
```
## 4. 닫힘 Closing 연산
```
기능: 팽창 후 침식을 수행하는 연산
용도: 객채 내부의 작은 구멍을 메우고, 객체의 경계를 부드럽게 만듦
객체의 경계선에 작은 틈이 있는 경우, 이를 채워 경계를 정리하는데 유용

closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
# cv2.MORPH_CLOSE 닫힘 연산 지정하는 플래그
```
### 열림: 침식 후 팽창 수행해 작은 노이즈 제거
### 닫힘: 팽창 후 침식 수행해 객체 내부의 작은 구멍 메워지고 경계 부드러워짐
![image](https://github.com/user-attachments/assets/7f0d950d-fc0b-43f4-95a6-6799aded81f4)



# 9. 슈퍼픽셀 알고리즘
```
슈퍼픽셀은 원본 이미지의 픽셀들을 유사한 특성을 기준으로 그룹화하여, 작은 영역으로 분할
SLIC Simple Linear Iterative Clustering 알고리즘은 이러한 슈퍼픽셀을 생성하는데 자주 사용

1. 슈퍼픽셀 Superpixel
이미지의 유사한 특성을 가진 픽셀들을 묶어 작은 영역으로 그룹화
개별 픽셀 대신 그룹화된 슈퍼픽셀을 사용하면 이미지의 구조를 더 효율적으로 표현하고 분석

2. SLIC Simple Linear Iterative Clustering
kmeans 군집화 방법에 기반해 슈퍼픽셀 생성하는 알고리즘
공간적 거리와 색상 유사성을 고려해 인접한 픽셀들이 유사할 때 동일한 슈퍼픽셀로 그룹화
반복적으로 클러스터 중심으로 업데이트, 유사한 특성을 가진 픽셀들이 최종적으로 동일한 슈퍼픽셀을 이루도록 함

3. SLIC 알고리즘 과정
초기 클러스터 중심 설정: 이미지에 일정 간격으로 초기 클러스터 중심 배치
픽셀 할당 및 거리 계산: 각 픽셀을 가까운 클러스터 중심에 할당, 공간적 거리와 색상 유사성 계산
클러스터 중심 업데이트: 각 클러스터의 중심을 업데이트하여, 슈퍼픽셀의 중심이 점차적으로 정확
=> 이 과정 여러번 반복해 안정된 슈퍼픽셀 형성

이미지 결과 >>
여러 육각형 형태의 슈퍼픽셀로 나뉨
이는 SLIC 알고리즘이 유사한 영역을 그룹화하여 생성된 결과
각 슈퍼픽셀은 색상과 위치가 비슷한 픽셀들이 모여서 하나의 영역 이룸 
```
![image](https://github.com/user-attachments/assets/a524234d-ec7d-46a7-95a1-b80bf6a096f3)
## SLIC 코드
```
import cv2

img = cv2.imread('lena.jpg')

slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=10.0)
# createSuperpixelSLIC 함수 사용해 SLIC 객체 생성
# region_size=20 슈퍼픽셀 크기, 값이 클수록 큰 영역의 슈퍼픽셀이 생성
# ruler=10.0 색상과 공간 간의 균형을 설정하는 파라미터, 값이 크면 색상보다는 공간적인 거리를 더 중요하게

slic.iterate(10) # 반복횟수 지정, 10번 반복해 슈퍼픽셀 점진적으로 최적화

mask_slic = slic.getLabelContourMask(thick_line=True)
# getLabelContourMask 사용해 슈퍼픽셀의 경계선 마스크 생성
thick_line=True 경계선 두껍게 표시

mask_slic_inv = cv2.bitwise_not(mask_slic)
# 경계선 마스크 색상 반전, 경계선 부분이 검정색이 되도록 변환해 원본 이미지와 결합하기 위함

image_result = cv2.bitwise_and(img, img, mask=mask_slic_inv)
# 반전된 경계선 마스크를 이용해 원본 이미지에서 슈퍼픽셀 경계가 있는 부분을 제거해 변수 저장
# 이미지 내의 경계 표시되지 않은 슈퍼픽셀 영역이 생성

image_contours = cv2.bitwise_and(img, img, mask=mask_slic)
# 경계선 마스크를 이용해 원본 이미지의 슈퍼픽셀 경계만 남김 결과를 contours 에 저장
# 슈퍼픽셀 경계선만 표시된 이미지가 생성

res = cv2.hconcat([img, image_contours, image_result]) # 가로로 합침 

cv2.imshow("Superpixel SLIC Result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

![image](https://github.com/user-attachments/assets/11e54812-cd16-454a-a173-f64c09bbb1a4)

# 9. 그랩컷 GrabCut
```
사용자가 제공하는 최소한의 정보를 바탕으로 이미지에서 객체와 배경을 분리하는 대화식 이미지 분할 알고리즘
그랩컷은 객체의 경계가 복잡하거나 배경이 복잡할 때도 효과적으로 객체 분리

1. 대화식 분할
그래브컷은 사용자가 분할 과정에 직접 참여하는 대화식 알고리즘
사용자는 마우스로 대략적인 경계를 지정하여 객체가 포함된 영역과 배경이 될 영역을 구분하는 초기 정보 제공

2. 작업 과정
사용자가 지정한 대략적인 경계를 바탕으로 알고리즘이 내부적으로 반복하여 객체와 배경을 구분
이 과정에서 배경과 객체의 픽셀 구분, 객체의 세부 경계를 정교하게 분할

3. 활용
객체가 복잡한 경계를 가지고 있거나, 배경이 복잡한 경우에도 효과적으로 객체를 분리
그랩컷은 사용자가 제공한 최소한의 정보로 최적의 분할 결과를 얻기 때문에 객체분리에 자주 사용

이미지 설명 >>
왼쪽: 사용자가 객체의 대략적인 경계 지정한 초기 상태
오른쪽: 그랩컷 알고리즘이 적용된 후 객체와 배경이 분리된 결과 이미
```
![image](https://github.com/user-attachments/assets/e44c0b36-6fc8-4ddb-a02d-d637a77157f2)
## 그랩컷 기능이 PPT에 내장
```
- PPT의 배경 제거 기능
파워포인트에서는 배경 제거 기능을 통해 사용자가 클릭 몇 번으로 이미지에서 불필요한 배경 삭제
-> 원하는 객체 남김
```
![image](https://github.com/user-attachments/assets/e8eef28a-4af1-4493-a871-e0cba26dd9b3)

## 그랩컷 알고리즘 적용하기 위해 초기설정 수행
```
import cv2
import numpy as np

img = cv2.imread('lena.jpg')
print(img.shape)

# 초기 마스크 생성
mask = np.zeros(img.shape[:2], np.uint8)
# 이미지와 동일한 크기의 마스크 배열 생성
# 초기 마스크는 모든 값이 0으로 설정, 각 픽셀이 배경 0 또는 전경 1인지 표시하는데 사용

# 배경과 전경 모델 생성
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
# 두 모델의 크기는 (1,65)로 설정
# 이 크기는 가중치1와 평균 벡터3, 공분산 행렬9, 가우시안 분포5 등 총 65개의 파라미터

# 전경 영역을 사각형으로 지정
rect = (50, 50, 150, 150)  # 임의로 설정한 전경 영역
rect_img = cv2.rectangle(img.copy(), (50, 50), (200, 200), (0, 255, 0), 2)
cv2.imshow("Rectangle", rect_img)
# cv2.rectangle 함수 사용해 전경 영역을 시각화, 이미지에 초록색 경계선 그

```

![image](https://github.com/user-attachments/assets/824b7284-4150-4e9a-9240-514739dd70a6)


## 그랩컷 알고리즘 사용해 이미지에서 배경과 전경 분리
```
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
# mask 그랩컷 실행 시 전경과 배경을 구분하는 마스크
# rect 초기 전경 영역을 설정한 사각형 좌표
# bgd_model과 fgd_model 배경과 전경 모델을 저장할 배열
# 5 는 그랩컷 반복 횟수
# cv2.GC_INIT_WITH_RECT 사각형 초기화를 사용해 전경과 배경 구분
# 이 함수는 전경과 배경을 구분하여 mask에 정보를 저장

# 마스크를 2D 이미지로 변환
mask_2d = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# mask에서 배경값이 0또는 2로 설정하고, 전경 값이(1또는3) 1로 설정, mask_2d 생성
# mask_2d는 전경과 배경을 구분하는 2D 마스크 변환

# 마스크로 원본 이미지에 적용
mask_color = mask_2d[:, :, np.newaxis] # np.newaxis 추가해 원본 이미지와 차원 맞춤
img_result = img * mask_color # 전경 부분만 남도록 img와 곱셉 수행해 img_result 생성

cv2.imshow('Original', img)
cv2.imshow('GrabCut', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
![image](https://github.com/user-attachments/assets/4a25923a-d9c5-4089-b713-9120c99feff2)




