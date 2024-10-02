## 💻 파이썬 💻
### 1️⃣
```
formatted_number = f"{random_number:06d}"
print(formatted_number)  # 출력: 6자리로 맞추기
```
```
1. : -> 포맷팅 지시어의 시작, 숫자를 어떻게 배열할지 지정
2. 06d -> 빈자리를 0으로 채우라는 의미, d는 decimal로 정수형 숫자 적용
```
### 2️⃣ 
```
print(f"주민등록번호: {front-back}")
```
```
f"{}"은 f-String 결과가 문자열에 직접 삽입되어 출력하는 역할
포맷 문자열이라는 기능을 사용해 문자열 내에 변수를 삽입
중괄호로 감싸지 않은 텍스트는 일반 문자열로 처리
각 변수는 중괄호로 감싸서 표현
1. f -> 해당 문자열이 f-String임을 나타냄
2. {} -> 해당 변수의 값이 문자열 내에 삽입
```
```
import random # 파이썬의 랜덤 모듈, 로또 번호를 무작위로 생성
```
```
def my_lotto_number():
    return sorted(random.sample(range(1, 46), 6))
# 1부터 45까지의 숫자 중 중복없이 6개의 숫자를 무작위로 선택,
선택한 숫자들을 오름차순으로 정렬하여 반환
```
```
def lotto_result(selected_number, correct_number):
    print("\n[로또 발표]")
    print(",".join(f"{num:02d}" for num in correct_number))
# 두 개의 인자를 받음(사용자가 구매한 번호, 로또 당첨 번호)
함수는 당첨 번호를 출력하고, 사용자의 번호들과 비교하여 몇 개가 일치하는지 출력
```
### 🔴 "구분자".join(문자열 리스트)
```
구분자: 각 문자열 사이에 들어갈 문자, 공백, 쉼표, 대시 등
문자열 리스트: 합치고 싶은 문자열이나 리스트, 튜플
# 리스트 안에 있는 숫자들을 쉼표로 구분해 하나의 문자열로 연결함
=> 리스트 안의 각각의 숫자 num을 두 자리 숫자로 포맷해 문자열로 변환
```
```
print("\n[내 로또 결과]")
for i, lotto in enumerate(selected_number):
    match_count = len(set(lotto) & set(correct_number))
    print(f"{chr(65 + i)} {','.join(f'{num:02d}' for num in lotto)} => {match_count}개 일치")
```
### 🔴 enumerate(): 리스트에서 인덱스와 값을 동시에 반환하는 함수
```
ex) enumerate(iterable, start=0)
⚫ iterable: 리스트, 튜플, 문자열 같은 반복가능한 객체
⚫ start: 인덱스 시작값을 지정, 기본값 0
```
```
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(i, fruit)
```
```
출력>
0 apple
1 banana
2 cherry
```
### for문에서의 f-String 쓰임
```
selected_number = [
    [10, 15, 19, 33, 34, 40],  # 첫 번째 로또 번호
    [20, 22, 23, 29, 42, 43],  # 두 번째 로또 번호
    [10, 14, 29, 30, 36, 44]   # 세 번째 로또 번호
]

for i, lotto in enumerate(selected_number):
    print(f"인덱스: {i}, 로또 번호: {lotto}")
```
```
인덱스: 0, 로또 번호: [10, 15, 19, 33, 34, 40]
인덱스: 1, 로또 번호: [20, 22, 23, 29, 42, 43]
인덱스: 2, 로또 번호: [10, 14, 29, 30, 36, 44]
```
### for 문에서 변수가 두 개인 이유?
```
for 문에서는 i와 lotto라는 두 변수를 사용하여 반복문을 실행
i: enumerate() 함수가 반환하는 인덱스
이 값은 나중에 chr(65+i)로 사용되어 A,B,C와 같은 문자라벨을 만듦
lotto: enumerate() 함수가 반환하는 리스트
selected_number 리스트에서 각 항목(사용자가 구매한 번호 리스트)
```
### 🔴 set(): 리스트를 집합(set)으로 변환
```
집합: 중복허용 X, 집합 연산할 때 유용하게 사용
&: 교집합 연산자, 두 집합 간의 공통으로 존재하는 원소 반환
len(): 교집합의 원소 개수 반환, 몇 개의 숫자가 일치하는지 계산
chr(): 주어진 숫자에 해당하는 ASCII 문자 반환, 65=A, 로또의 라벨
```
```
selected_number = [my_lotto_number() for _ in range(number)]
```
```
my_lotto_number(): 1부터 45까지 숫자 중 중복없이 6개의 숫자를 '리스트'
sorted(): 리스트를 오름차순으로 정렬, 결과는 리스트
selected_number: 여러 개의 리스트를 담은 리스트로 생성
_ (언더스코어): 변수의 값을 사용하지 않을 때 사용, 단지 반복 횟수를 나타
리스트 컴프리헨션: my_lotto_number() 함수가 여러 번 호출된 결과를 리스트로 저장,
number 만큼 my_lotto_number() 호출하여, 각 호출에서 반환된 리스트를 selected_number에 담음
```
```
<예시 코드>
for _ in range(5):
    print("Hello, world!")
=> 단지 5번반복하는 데에만 사용되며, 0~4까지의 값을 이용하지 않음
```
```
print("\n[구매한 로또 번호]")
for i, lotto in enumerate(selected_number):
    print(f"{chr(65+i)} {','.join(f'{num:02d}' for num in lotto)}")
```
```
for 루프: selected_number 리스트에 있는 각 로또 번호에 대해 반복 작업
enumerate(): 리스트에서 인덱스 번호와 요소를 '튜플로 반환'
=> 각 요소의 인덱스 번호와 각각의 로또 번호 리스트를 함께 반환
chr(): '아스키 코드값'에 해당하는 문자를 반환하는 함수
리스트 컴프리헨션: lotto 리스트에 들어있는 각 번호 num을 대상으로 특정한 형식으로 반환
=> 각 로또 번호를 2자리로 맞춰서 출
```
```
correct_number = my_lotto_number()
lotto_result(selected_number, correct_number)
lotto()

```




