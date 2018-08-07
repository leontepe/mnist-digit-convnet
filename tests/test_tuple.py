import unittest

class TupleTest(unittest.TestCase):

    def test_add_tuples(self):
        a = (3, 5, 6)
        b = 2 + a
        self.assertEqual(b, (2, 3, 5, 6))

if __name__ == '__name__':
    unittest.main()