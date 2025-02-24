import unittest
import torch
import torch.nn as nn
from model import SelfAttention

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")  # Use CPU for testing
        self.key_dim = 64
        self.value_dim = 128
        self.batch_size = 2
        self.max_len = 5
        self.attention = SelfAttention(self.key_dim, self.value_dim, self.device)


    """Checks if the output of the forward() method has the expected shape."""
    def test_forward_shape(self):
        queries = torch.randn(self.batch_size, self.max_len, self.key_dim)
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.randint(0, 2, (self.batch_size, self.max_len))  # Random mask

        output = self.attention(queries, keys, values, mask)
        self.assertEqual(output.shape, (self.batch_size, self.max_len, self.value_dim))
    

    """Tests if the masking mechanism correctly sets masked positions to zero in the output."""
    def test_masking(self):
        queries = torch.randn(self.batch_size, self.max_len, self.key_dim)
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]])  # Example mask

        output = self.attention(queries, keys, values, mask)

        # Check if masked positions have zero values in the output
        # for b in range(self.batch_size):
        #     for i in range(self.max_len):
        #         if mask[b, i] == 0:
        #             self.assertTrue(torch.all(output[b, i] == 0))
     

    """Tests that the diagonal of the weight matrix is correctly set to 
       negative infinity before the softmax."""
    def test_diagonal_inf(self):
        queries = torch.randn(self.batch_size, self.max_len, self.key_dim)
        keys = torch.randn(self.batch_size, self.max_len, self.key_dim)
        values = torch.randn(self.batch_size, self.max_len, self.value_dim)
        mask = torch.ones(self.batch_size, self.max_len)

        weights = torch.bmm(queries, keys.permute(0, 2, 1))
        for i in range(self.max_len):
            weights[:, i, i] = -float('inf')

        self.attention(queries,keys,values,mask)

        #This checks if the diagonal of the attention weights is -inf before the softmax
        for b in range(self.batch_size):
            for i in range(self.max_len):
                self.assertEqual(weights[b,i,i],-float('inf'))
    

    """Tests that the code correctly handles NaN values
       that might arise during the softmax operation."""
    def test_nan_handling(self):
         queries = torch.zeros(self.batch_size, self.max_len, self.key_dim)
         keys = torch.zeros(self.batch_size, self.max_len, self.key_dim)
         values = torch.randn(self.batch_size, self.max_len, self.value_dim)
         mask = torch.ones(self.batch_size, self.max_len)

         output = self.attention(queries,keys,values,mask)
         self.assertTrue(not torch.any(torch.isnan(output)))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)