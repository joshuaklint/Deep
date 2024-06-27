import unittest
from employee import Employee
class EmployeeTest(unittest.TestCase):
    def setUp(self):
        self.e1 = Employee('Joshua', 'Agbanu', 1000)
        self.e2 = Employee('Collins','Gameli', 2000)
    def tearDown(self):
        pass
    def test_email(self):

        self.assertEqual(self.e1.emails, 'joshuaagbanu@gmail.com')
        self.assertEqual(self.e2.emails, 'collinsgameli@gmail.com')

