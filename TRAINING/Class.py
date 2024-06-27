#Tricks with conditions

condition =True
if condition:
    x = 1
else:
    x = 0
print(x)

x = 1 if condition else 0
print(x)

num1 = 10_000_000_000
num2 = 100_000_000
total = num1 + num2
print(f'{total:,}')

#Common bad practices
with open('../Deep Learning/Example.txt', 'r') as f:
    lines = f.readlines(10)
    print(lines)

clubs = ['Club 1', 'Club 2', 'club 3', 'club 4', 'club 5']
for index, names in enumerate(clubs, start=1):
    print(f'{index}: {names}')

#Using of ZIP METHOD
#oTHER WAYS
names = ['Peter Parker', 'Clark Kent','Wade Wilson', 'Jack Wilson']
heroes = ['Spiderman', 'Deadpool','Superman','Batman']
universe = ['Marvel', 'DC','Marvel','DC']

for index, name in enumerate(names):
    hero = heroes[index]
    print(f'{index}  { name} is actually {hero} ')

#THE TRICK (ZIP)
for name, hero,uni in zip(names, heroes, universe):
    print(f'{name} is actually {hero} from {uni}')

for value in zip(names, heroes, universe):
    print(f'{value} is actually the tuple')


#UNPACKING
a, b, *c,d= (1,2,3,4,5,6,7,8,9,10)
print(a,b,c,d)

a,b,*_,d= (1,2,3,4,5,6,7,8,9,10)
print(a,b,d)





