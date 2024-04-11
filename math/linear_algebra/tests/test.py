import unittest
from the_whole_barn import add_matrices

class testAddMatrices(unittest.TestCase):
    def test_add_matrices_mismatched_shapes(self):
        mat1 = [[[[1]], [[2, 3]]], [[[4, 5, 6]]]]
        mat2 = [[[[1]], [[2, 3]]], [[[4, 5, 6]]]]
        result = add_matrices(mat1, mat2)
        self.assertIsNone(result, "Expected None for mismatched shapes")

if __name__ == '__main__':
    unittest.main()
