import unittest
from fashion.lib import Fashion

class TestFashion(unittest.TestCase):
    def setUp(self):
        self.fashion = Fashion()
        self.fashion.load_data()
        self.fashion.transform_data()
        
    def test_load_data(self):
        self.assertIsNotNone(self.fashion.X_train)
        self.assertIsNotNone(self.fashion.X_test)
        self.assertIsNotNone(self.fashion.y_train)
        self.assertIsNotNone(self.fashion.y_test)

    def test_transform_data(self):
        self.assertEqual(self.fashion.X_train.shape, (48000, 28, 28, 1))
        self.assertEqual(self.fashion.X_test.shape, (12000, 28, 28, 1))
        self.assertEqual(self.fashion.y_train.shape, (48000, 10))
        self.assertEqual(self.fashion.y_test.shape, (12000, 10))

    def test_get_data(self):
        X_train, X_test, y_train, y_test = self.fashion.get_data()
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_build_model(self):
        model = self.fashion.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.output_shape, (None, 10))