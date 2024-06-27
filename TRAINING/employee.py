class Employee:

    raise_amt = 1.05
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
    @property
    def emails(self):
        return f"{self.first}{self.last}@gmail.com".lower()
    @property
    def fullnames(self):
        return f"{self.first} {self.last}".title()
    @property
    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)
        return self.pay

#emp1 = employee("Joshua", "Smith", 100000)
#print(emp1.email())
#print(emp1.fullname())
#print(emp1.apply_raise())