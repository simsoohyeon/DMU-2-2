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
### "구분자".join(문자열 리스트)
```
구분자: 각 문자열 사이에 들어갈 문자, 공백, 쉼표, 대시 등
문자열 리스트: 합치고 싶은 문자열이나 리스트, 튜플
# 리스트 안에 있는 숫자들을 쉼표로 구분해 하나의 문자열로 연결함
리스트 안의 각각의 숫자 num을 두 자리 숫자로 포맷해 문자열로 변환
```
```
print("\n[내 로또 결과]")
for i, lotto in enumerate(selected_number):
    match_count = len(set(lotto) & set(correct_number))
    print(f"{chr(65 + i)} {','.join(f'{num:02d}' for num in lotto)} => {match_count}개 일치")
```
### enumerate() 
```
selected_number = [my_lotto_number() for _ in range(number)]

```
```
print("\n[구매한 로또 번호]")
for i, lotto in enumerate(selected_number):
    print(f"{chr(65+i)} {','.join(f'{num:02d}' for num in lotto)}")
```
```
correct_number = my_lotto_number()
lotto_result(selected_number, correct_number)
lotto()

```




