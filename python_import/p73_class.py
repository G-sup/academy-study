class Person:
    def __init__(self,name,age,address) :   # init 는 반드시 정의 되어야 하며 self 도 반드시 있어야한다
        self.name = name
        self.age = age
        self. address = address

    
    def greeting(self):   # class 안에 있는 변수는 self를 무조건 넣어준다
        print("안녕하세요, 저는 {0}입니다.".format(self.name))


 