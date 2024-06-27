import unittest
import Calc

class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(Calc.add(10,5),15)
        self.assertEqual(Calc.add(0, 5), 5)
        self.assertEqual(Calc.add(1, 5), 6)
    def test_sub(self):
        self.assertEqual(Calc.subtract(10, 5), 5)
        self.assertEqual(Calc.subtract(0, 5), -5)
        self.assertEqual(Calc.subtract(1, 5), -4)

    def test_multiply(self):
        self.assertEqual(Calc.multiply(10, 5), 50)
        self.assertEqual(Calc.multiply(0, 5), 0)
        self.assertEqual(Calc.multiply(1, 5), 5)

    def test_divide(self):
        self.assertEqual(Calc.divide(10, 5), 2)
        self.assertEqual(Calc.divide(1, 5), 0.2)
        self.assertEqual(Calc.divide( 10, 4), 2.5)

        with self.assertRaises(ZeroDivisionError):
            Calc.divide(10,0)

if __name__ == '__main__':
    unittest.main()
