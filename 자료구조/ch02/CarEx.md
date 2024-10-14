# Car 클래스 정의
```
class Car :
    def __init__(self, color, speed = 0) : # Car 객체가 생성될 때 호출
    # color와 speed를 매개변수를 받아 객체의 속성으로 초기화,
    # speed의 기본값이 0이므로, 인스턴스가 생성될 때 값을 주지 않으면 속도 0으로 설정
        self.color = color
        self.speed = speed

    def speedUp(self) : # self.speed는 현재 객체의 속도 의미
        self.speed += 10

    def speedDown(self) :
        self.speed -= 10

    def __lt__(self, carB) : # __lt__() 메소드는 작은지 < 비교
       return self.speed < carB.speed

    def __gt__(self, carB) : # __gt__() 메소드는 큰지 > 비
       return self.speed > carB.speed

    def __str__(self) : # 객체 정보를 문자열로 출력 
        return "color = %s, speed = %d" % (self.color, self.speed)

```
# 메인함수 
```
if __name__ == "__main__":
    car1 = Car('black', 0)			# 검정색, 속도 0
    car2 = Car('red', 120)			# 빨간색, 속도 120

    print("car1:", car1) # __str__() 메소드에 의해 정보 출력 
    print("car1:", car2) # color와 speed 정보 출력 

    print("car1 < car2 : ", car1 < car2)
    print("car1 > car2 : ", car1 > car2)
```
## 🔵 오버로딩 메소드
```
오버로딩이란, 같은 이름의 메소드나 연산자가 다양한 방식으로 동작하도록 정의
객체 지향 프로그래밍에서 연산자나 메서드를 오버로딩하면, 연산자나 함수 호출을 객체에 맞게 재정의
```

## 🔵 if __name__ == "__main__":
```
파이썬 코드가 직접 실행되었을 때 특정 코드 블록이 실행되도록 하기 위한 구문
파일이 모듈로 임포트될 때, 해당 블록을 실행하지 않도록 하여, 코드의 재사용성 높임
```
## 🔵 __init__() 메소드
```
__init__()는 파이썬 클래스의 생성자로, 객체가 생성될 때 자동으로 호출되는 초기화 메소드
이 메소드를 통해 클래스가 객체로 인스턴스화될 때, 객체의 초기상태 설정하는 역할
생성자는 객체가 생성될 때 필요한 초기값을 설정하거나, 객체의 초기상태 정의하는 함수
self: 항상 첫 번째 매개변수로 전달, 새로 생성된 객체 자신 참조 
```
## 🔵 2.12 상속
```
상속은 객체지향 프로그래밍의 중요한 개념 중 하나로, 기존 클래스(부모 또는 슈퍼)의 속성과 메소드를
새로운 클래스(자식 또는 서브)가 물려받는 것을 말함 상속을 통해 코드의 재사용성을 높이고,
기존 클래스를 확장하거나 수정하는 방식으로 새로운 기능을 추가할 수 있음
상속을 통해 자식 클래스는 부모 클래스의 모든 속성과 메소드를 자동으로 사용할 수 있으며,
추가적인 속성이나 메소드를 정의할 수 있음 부모클래스의 메소드를 오버라이딩(재정의)하여
자식 클래스에서 새롭게 동작할 수 있도록 변경할 수도 있음 
```
```
부모 클래스: 자식 클래스가 상속받는 클래스로, 기존의 기능 포함
자식 클래스: 부모 클래스로부터 상속을 받아 새로운 기능을 추가하거나 기존 기능 수정
```
###  상속 예시 코드
```
# 부모 클래스 정의
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."


# 자식 클래스 정의 - Animal 클래스를 상속받음
class Dog(Animal):
    def speak(self):
        return f"{self.name} barks."


# 자식 클래스 정의 - Animal 클래스를 상속받음
class Cat(Animal):
    def speak(self):
        return f"{self.name} meows."


# 객체 생성 및 사용
dog = Dog("Buddy")  # Dog 클래스의 객체 생성
cat = Cat("Whiskers")  # Cat 클래스의 객체 생성

# 메서드 호출
print(dog.speak())  # 출력: Buddy barks.
print(cat.speak())  # 출력: Whiskers meows.
```
### 부모 클래스
```
- Animal 클래스는 모든 동물의 공통 속성인 name을 받아 객체를 초기화
- speak() 메소드를 정의하여 동물이 소리내는 일반적인 동작 구현
```
### 자식 클래스
```
- Dog 클래스와 Cat 클래스는 Animal 클래스 상속받음
즉, Animal 클래스의 __init__() 메소드를 그대로 사용하여 이름을 설정
- 각 클래스는 자신만의 소리를 내기 위해 부모 클래스 speak() 메소드를 오버라이딩 
```
### 객체 생성
```
Dog과 Cat 클래스는 각각 상속받은 Animal 클래스의 초기화 방법 사용, 이름 설정하며 생성
```
