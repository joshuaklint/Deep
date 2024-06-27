'''student = {'name':'John', 'age':22, 'courses':['Math', 'English']}
student['phone'] = '01-123-4567'
print(student)
print(student.get('name'))

#Update dictionary
student.update({'name':'joshua','age':27})
print(student)
print(len(student))
print(student.keys())
print(student.values())
print(student.items())

for keys, values in student.items():
    print(keys,values)'''

import turtle
wn = turtle.Screen()
alex = turtle.Turtle
alex.forward(170)
alex.left(90)
alex.forward(75)