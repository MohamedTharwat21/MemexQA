import unittest
import torch
import torch.nn as nn
import numpy as np
from model import FVTA

class TestFVTA(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.key_dim = 32
        self.value_dim = 64
        self.batch_size = 16
        self.max_query_len = 12
        self.max_len = 140
        self.fvta = FVTA(self.key_dim, self.value_dim, self.device)
    
    """Verifies that the output shape is correct: (batch_size, value_dim)."""
    def test_forward_shape(self):
        query = torch.randn(self.batch_size, self.max_query_len, self.key_dim)
        query_lengths = torch.randint(1, self.max_query_len + 1, (self.batch_size,))
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.randint(0, 2, (self.batch_size, self.max_len))

        output = self.fvta(query, query_lengths, keys, values, mask)
        self.assertEqual(output.shape, (self.batch_size, self.value_dim))
    
    """Checks if the query mask is applied correctly.
       It verifies that the query elements beyond the query_lengths are set to zero."""
    # def test_masking_query(self):
    #     query = torch.randn(self.batch_size, self.max_query_len, self.key_dim)
    #     query_lengths = torch.tensor([3, 5, 7, 9, 11, 1, 2, 4, 6, 8, 10, 12, 3, 5, 7, 9])
    #     keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
    #     values = torch.randn(self.batch_size, self.max_len, self.value_dim)
    #     mask = torch.randint(0, 2, (self.batch_size, self.max_len))

    #     output = self.fvta(query.clone(), query_lengths, keys, values, mask)

    #     for b in range(self.batch_size):
    #         for i in range(query_lengths[b], self.max_query_len):
    #             self.assertTrue(torch.all(query[b, i] == 0.0))

    # """Verifies that the keys and values are masked correctly according to the provided mask."""
    # def test_masking_keys_values(self):
    #     query = torch.randn(self.batch_size, self.max_query_len, self.key_dim)
    #     query_lengths = torch.randint(1, self.max_query_len + 1, (self.batch_size,))
    #     keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
    #     values = torch.randn(self.batch_size, self.max_len, self.value_dim)
    #     mask = torch.tensor([[1, 1, 0, 0, 0] + [1] * (self.max_len - 5) for _ in range(self.batch_size)])

    #     output = self.fvta(query, query_lengths, keys.clone(), values.clone(), mask)

    #     for b in range(self.batch_size):
    #         for i in range(self.max_len):
    #             if mask[b, i] == 0:
    #                 self.assertTrue(torch.all(keys[b, i] == 0.0))
    #                 self.assertTrue(torch.all(values[b, i] == 0.0))
    
    """Ensures that the function handles potential NaN values correctly.
        This is important when dealing with attention weights and softmax."""
    def test_nan_handling(self):
        query = torch.zeros(self.batch_size, self.max_query_len, self.key_dim)
        query_lengths = torch.randint(1, self.max_query_len + 1, (self.batch_size,))
        keys = torch.zeros(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.ones(self.batch_size, self.max_len)

        output = self.fvta(query, query_lengths, keys, values, mask)
        self.assertTrue(not torch.any(torch.isnan(output)))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)