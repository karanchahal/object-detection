import unittest

class TestAnchors(unittest.TestCase):

    def test_iou(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_offsets(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_anchor_generation(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()