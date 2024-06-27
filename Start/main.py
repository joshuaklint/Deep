class Animals():

    def __init__(self,name, age):
        self.name = name
        self.age = age

    def speak(self):
        print('The ', self.age, 'year old', self.name,'SHOUT' )


class Dog(Animals):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color
    def speak(self):
        print('The', self.color, self.age, 'year old', self.name, 'Barks')

class Cat(Dog):
    def __init__(self, name, age, color, attitude):
        super().__init__(name, age, color)
        self.attitude = attitude

    def behaviour(self):
        print('The', self.attitude, self.color, self.age, 'year old', self.name, 'mourn meaow')

cat = Cat('CR7', 3, 'White', 'Clever')
cat.behaviour()
cat.speak()

dog = Dog('Suarez', 4, 'Black')
dog.speak()
