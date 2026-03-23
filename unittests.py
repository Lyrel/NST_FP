import unittest
import torch
import numpy as np

# Import the class from assessment.py
from assessment import Assessment

class TestNSTModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test"""
        self.test_tensor = torch.rand(1, 3, 224, 224)
        self.assessment = Assessment()

    def test_tensor_to_numpy_type(self):
        """Test that function returns numpy array"""
        result = self.assessment.tensor_to_numpy(self.test_tensor)
        self.assertIsInstance(result, np.ndarray)

    def test_tensor_to_numpy_dtype(self):
        """Test that numpy array has correct dtype"""
        result = self.assessment.tensor_to_numpy(self.test_tensor)
        self.assertEqual(result.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()