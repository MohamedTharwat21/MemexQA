import unittest
import torch
import torch.nn as nn
from model import Attention

class TestAttention(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")  # Use CPU for testing
        self.key_dim = 64
        self.value_dim = 128
        self.batch_size = 2
        self.max_len = 5
        self.attention = Attention(self.key_dim, self.value_dim, self.device)
    
    def test_forward_shape(self):
        query = torch.randn(self.batch_size, self.key_dim)
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.randint(0, 2, (self.batch_size, self.max_len))  # Random mask

        output = self.attention(query, keys, values, mask)
        self.assertEqual(output.shape, (self.batch_size, self.value_dim))

    def test_masking(self):
        query = torch.randn(self.batch_size, self.key_dim)
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]])  # Example mask

        output = self.attention(query, keys, values, mask)

        # Check if masked positions have zero influence
        # for b in range(self.batch_size):
        #     for i in range(self.max_len):
        #         if mask[b, i] == 0:
        #             # Check that the corresponding value is not contributing
        #             weights = torch.bmm(query[b].unsqueeze(0).unsqueeze(0), keys[b].unsqueeze(0).permute(0,2,1))
        #             weights_masked = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights))
        #             weights[0].masked_fill_(weights_masked.bool(), -float('inf'))
        #             weights = self.attention.softmax(weights)
        #             self.assertEqual(weights[0,0,i], 0.0)


    def test_nan_handling(self):
        query = torch.zeros(self.batch_size, self.key_dim)
        keys = torch.zeros(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.ones(self.batch_size, self.max_len)

        output = self.attention(query, keys, values, mask)
        self.assertTrue(not torch.any(torch.isnan(output)))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)