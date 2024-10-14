## Numpy와 Pytorch의 Tensor
### 1. Numpy
```
- 파이썬에서 고성능 수치 계산을 위한 라이브러리로, 특히 배열 연산 처리에 유용한 기능 제공
- 이 라이브러리는 데이터과학, 인공지능, 기계학습 등의 다양한 분야에서 사용
```
### 2. Tensor
```
- Tensor는 다차원 배열을 의미, Numpy배열과 유사하지만, GPU가속을 통해 더 빠른 연산 가능
- 텐서는 딥러닝에서 데이터를 표현하고 처리하는 기본 단위로, PyTorch에서 중요한 역할
```
### Numpy and Tensor 배열 코드
```
- Numpy 배열 생성: np.array([1,2,3]) -> 1차원 배열 생성
- PyTorch 텐서 생성: torch.tensor([1,2,3]) -> 텐서 생성, Numpy배열과 매우 유사한 형식으로 작동 
```
![image](https://github.com/user-attachments/assets/707e8556-2f28-40dd-af06-1997460e95ef)

### 브로드 캐스팅 Broadcasting
```
- 브로도캐스팅은 서로 다른 크기의 배열 간에도 연산을 수행할 수 있도록 해주는 기능
- 일반적으로 배열 연산은 같은 크기의 배열에서만 가능하지만, 브로드캐스팅을 사용하면 크기가
다른 배열 간에도 자동으로 크기를 맞춰 연산을 수행할 수 있음
- PyTorch의 텐서는 Numpy와 동일한 브로드캐스팅 규칙을 따름
```
### Numpy에서의 브로도 캐스팅
```
import numpy as np
np_array = np.array([1, 2, 3]) 
np_broadcast = np_array + 1  # [2, 3, 4]
```
### PyTorch에서 동일한 연산
```
import torch
torch_tensor = torch.tensor([1, 2, 3])
torch_broadcast = torch_tensor + 1  # [2, 3, 4]
```
![image](https://github.com/user-attachments/assets/c7d91c9a-1ff2-42e7-a90d-c3418025ee47)
### 차원과 축 조작
### 1. 차원 Dimension
```
- 차원은 다차원 배열에서 각 배열이 몇 차원인지 나타내는 개념
- Numpy에서는 ndim 속성을 사용하여 배열의 차원을 확인할 수 있음
ex) 2차원 배열은 행과 열을 기준으로 값을 저장
```
### 2. 축 Axis
```
- 축은 배열이나 텐서의 특정 방향을 나타내며, 연산을 할 때 축을 기준으로 합계를 구함
- 2차원 배열의 경우, 축0은 열별로 연산결과 반환(세로), 축1은 행별로 연산결과 반환(가로) 
ex) 각 열의 합 또는 각 행의 합 구할 수 있음
```
### Numpy에서의 차원과 축 조작
```
import numpy as np
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print(np_array.ndim)  # 2차원 배열확인 

# 축을 기준으로 합계 구하기
sum_along_axis0 = np.sum(np_array, axis=0)  # 세로: 각 열의 합 [5, 7, 9]
sum_along_axis1 = np.sum(np_array, axis=1)  # 가로: 각 행의 합 [6, 15]
```
### PyTorch에서의 차원과 축 조작
```
import torch
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(torch_tensor.dim())  # 2차원 텐서

# 축을 기준으로 합계 구하기
sum_along_axis0_torch = torch.sum(torch_tensor, dim=0)  # 각 열의 합 [5, 7, 9]
sum_along_axis1_torch = torch.sum(torch_tensor, dim=1)  # 각 행의 합 [6, 15]
```
![image](https://github.com/user-attachments/assets/f9234ae0-51d9-435b-8c53-423e6f4ac36d)
### 배열 슬라이싱 및 인덱싱 
### 1. 슬라이싱 Slicing
```
- 슬라이싱은 배열의 일부를 선택하여 새로운 배열로 만드는 방법
- 파이썬의 리스트처럼 배열에서도 슬라이싱을 사용할 수 있음, 특정 부분의 데이터 쉽게 추출
```
### 2. 인덱싱 Indexing
```
- 인덱싱은 배열에서 특정 위치에 있는 값을 선택하는 방법
인덱스를 사용하여 배열의 특정 요소에 접근하거나 부분 배열 선택 가능 
```
### Numpy 배열 슬라이싱
```
import numpy as np
np_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
slice_np = np_array[1:, :2]  # 두 번째 행부터 첫 번째, 두 번째 열까지 선택
```
### PyTorch 텐서 슬라이싱
```
import torch
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
slice_torch = torch_tensor[1:, :2]  # 두 번째 행부터 첫 번째, 두 번째 열까지 선택
```
### 결과 [[4, 5], [7, 8]]
![image](https://github.com/user-attachments/assets/663003eb-bb9e-44f6-aa7b-9ca433783f1c)

### 배열 연산
```
- Numpy에서는 배열 간 사칙연산 등 다양한 연산 지원하며,
PyTorch에서도 동일한 방식으로 배열=텐서 간 연산을 수행할 수 있음
=> 이러한 요소별 연산은 딥러닝 모델 구현 시 자주 사용되며,
두 배열 또는 텐서의 같은 위치에 있는 원소끼리 연산 수행 
```
### Numpy에서의 배열 연산
```
import numpy as np

# 두 개의 1차원 배열 생성
np_array1 = np.array([1, 2, 3])
np_array2 = np.array([4, 5, 6])

# 두 배열의 요소별 덧셈 연산
sum_np = np_array1 + np_array2  # [5, 7, 9]
```
### PyTorch에서의 텐서 연산
```
import torch

# 두 개의 1차원 텐서 생성
torch_tensor1 = torch.tensor([1, 2, 3])
torch_tensor2 = torch.tensor([4, 5, 6])

# 두 텐서의 요소별 덧셈 연산
sum_torch = torch_tensor1 + torch_tensor2  # [5, 7, 9]
```
![image](https://github.com/user-attachments/assets/93e54554-8046-466a-859e-f2dc5a7a7701)
### 배열 변형 및 재구성 Reshaping
```
Reshape는 배열의 차원을 변경하여 원하는 모양으로 데이터를 재구성할 수 있게 해주는 함수
이 기능은 데이터 전처리와 딥러닝 모델 구현 시 매우 중요한 역할,
데이터를 적절한 크기로 변형하여 입력으로 사용 
```
### Numpy에서의 배열 변형
```
import numpy as np

# 1차원 배열 생성
np_array = np.array([1, 2, 3, 4, 5, 6])

# (2, 3) 배열로 변형
reshaped_np = np_array.reshape((2, 3))  # [[1, 2, 3], [4, 5, 6]] 2X3 행X열
```
### PyTorch에서의 텐서 변형
```
import torch

# 1차원 텐서 생성
torch_tensor = torch.tensor([1, 2, 3, 4, 5, 6])

# (2, 3) 텐서로 변형
reshaped_torch = torch_tensor.view(2, 3)  # [[1, 2, 3], [4, 5, 6]]
```
### 랜덤 데이터 생성
```
랜덤 데이터 생성은 신경망 학습 시 매우 중요한 역할, 가중치 초기화에 랜덤 값 사용할 때가 많음
- 신경망에서 가중치를 학습시킬 때 초기 가중치를 랜덤 값으로 설정하는 것이 중요
- 무작위 초기화는 최적화 알고리즘이 올바르게 작동하고, 특정한 패턴에 의존하지 않게 해줌
```
### Numpy에서 랜덤 데이터 생성
```
import numpy as np

# 3x3 크기의 랜덤 값으로 이루어진 배열 생성
random_np = np.random.rand(3, 3)
```
## PyTorch에서 랜덤 데이터 생성
```
import numpy as np

# 3x3 크기의 랜덤 값으로 이루어진 배열 생성
random_np = np.random.rand(3, 3)
```

## 프로그래밍 실습 문제 1 
```
• 크기는 3×4, Random 값을 가진 텐서(Tensor)를 생성
• 생성된 텐서의 두 번째 열을 모두 0으로 변경
• 텐서의 모든 원소를 합한 값을 계산하여 print
```
```
import torch

tensor = torch.rand(3,4) # 배열 생성, 각 원소는 0과 1 사이 무작위 값

tensor[:,1] = 0 # 두 번째 열 모두 0로 설정 

sum_all_elements = torch.sum(tensor) # 텐서의 모든 원소를 더한 값 반환 

print("Tensor:\n", tensor)
print("Sum of all elements:", sum_all_elements.item())
# .item()은 텐서가 오직 하나의 값을 가지고 있을 때 사용, 파이썬의 자료형으로 변환하여 반환
```
### 프로그래밍 실습 문제 2
```
• 크기는 5×5, Random 값을 가진 텐서(Tensor)를 생성
• 생성된 텐서에서 값이 0.5보다 큰 경우만 추출하여 1차원 텐서로 출력하기
```
```
import torch

tensor = torch.rand(5,5)

filtered_tensor = tensor[tensor>0.5] # 0.5보다 큰 원소 추출, 1차원 텐서로 반

print("orginal tensor:\n", tensor)
print("filtered tensor (values > 0.5)\n", filtered_tensor)
```
### 프로그래밍 실습 문제 3
```
• 크기는 4×4, Random 값을 가진 Numpy 배열(Array)을 생성
• 생성된 배열을 Tensor로 변환하고 파일로 저장하기
• 저장된 파일을 불러와 원본 텐서와 동일한지 확인 (비교함수 적용)
```
```
import numpy as np
import torch

# 1. 크기 4x4의 NumPy 배열 생성
np_array = np.random.rand(4, 4)
print("Original NumPy Array:\n", np_array)

# 2. NumPy 배열을 PyTorch 텐서로 변환
torch_tensor = torch.from_numpy(np_array) # torch.from_numpy()는 텐서 변환 함
print("Converted PyTorch Tensor:\n", torch_tensor)

# 3. PyTorch 텐서를 'torch_tensor.pt' 파일로 저장
torch.save(torch_tensor, 'torch_tensor.pt')

# 4. 저장된 파일을 불러와 텐서로 변환
loaded_tensor = torch.load('torch_tensor.pt')
print("Loaded PyTorch Tensor:\n", loaded_tensor)

# 5. 원본 텐서와 불러온 텐서가 동일한지 확인
print("Are they equal?", torch.equal(torch_tensor, loaded_tensor))
```
### 코드 문법
```
1. Numpy 배열 생성 np.random.rand(4,4)
2. PyTorch 텐서로 변환 torch.from_numpy() # numpy -> tensor
3. 파일로 저장 torch.save(텐서이름, 파일명.pt)
4. 파일에서 불러오기 torch.load(파일명.pt)
5. 동일성 확인 torch.equal(A,B) 
```









































