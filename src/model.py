"""
# may be helpful :
Some tricks (especially for multi-head attention and positional encoding) and decoding Language Models
https://atcold.github.io/NYU-DLSP20/en/week12/12-1/#:~:text=maximum%20sequence%20length-,Some%20tricks%20(especially%20for%20multi%2Dhead%20attention%20and%20positional%20encoding)%20and,Really%20helpful%20for%20a%20task%20like%20machine%20translation,-The%20following%20are
"""

import torch
import numpy as np
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
SelfAttention v.s. Attention classes v.s. FVT Attention:

Input shape:
    SelfAttention:(batch_size, max_len, dim) for queries, keys, and values.
    
    Attention (Cross attention ): /"query/" is a single vector (batch_size, key_dim),
               keys and values are sequences (batch_size, max_len, dim).

Masking:
    Both classes use masks to ignore certain positions in the sequence (e.g., padding tokens).
    Masks are applied to keys and values to set specific elements to 0.

Weight Computation:
    SelfAttention: Computes a full attention weight matrix (batch_size, max_len, max_len)
    ,where each token attends to all others in the sequence.
    
    Attention: Computes a simpler attention weight matrix (batch_size, 1, max_len)
    ,where the single query attends to the entire sequence.

Output:
    SelfAttention: Produces contextualized embeddings for all tokens in the sequence.
    SA Output Dim: (batch_size, max_len, value_dim) (64 , 6 , 512)
    Attention: Produces a single contextualized embedding for the input query.
    Att Output Dim: (batch_size, value_dim).

"""



class SelfAttention(nn.Module):
    """key_dim will be used for Query_dim also  : Q.K^T """
    
    """if you used .bool() for the masks there will be error in masked_fill_ 
       you have to use .bool() instead .
       the dafault for the mask is .bool() ,
       so you must add .bool(). """
    def __init__(self, key_dim, value_dim, device):
        super(SelfAttention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1) #(batch_size ,rows,cols) for key matrix for example
        
    def forward(self, queries, keys, values, mask):
        """
        Attention mechanisms use this mask to prevent 
        padded positions or irrelevant parts of the 
        sequence from affecting the computation of
        attention weights.
        """
  
        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim).bool() # (batch_size, max_len, key_dim)
        # query_mask = mask.unsqueeze(-1).repeat(1, 1, self.query_dim).bool()  # (batch_size, max_len, key_dim) 
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim).bool() # (batch_size, max_len, value_dim)
        
        queries.data.masked_fill_(key_mask.bool(), 0.)
        keys.data.masked_fill_(key_mask.bool(), 0.)
        values.data.masked_fill_(value_mask.bool(), 0.)

        max_len = keys.shape[1]
        #weights = queries @ keys.transpose(-2,-1)
        weights = torch.bmm(queries, keys.permute(0,2,1)) # (batch_size, max_len, max_len)

        #weight_mask = (weights == 0.0).bool() 
        weight_mask = torch.where(weights == 0.,
                                  torch.ones_like(weights),
                                  torch.zeros_like(weights)).bool()
        weights.data.masked_fill_(weight_mask.bool(), -float('inf'))
        

        # Apply softmax and handle NaNs
        # make the diagonal also infinit 
        for i in range(max_len):
            # for all batches make the row i and col i == inf
            weights[:,i,i] = -float('inf')

        weights = self.softmax(weights)
        # weight_mask = torch.isnan(weights) [matrix of True's and False's]
        weight_mask = torch.where(torch.isnan(weights),
                                  torch.ones_like(weights),
                                  torch.zeros_like(weights)).bool()
        weights.data.masked_fill_(weight_mask.bool(), 0.)
        
        # bmm as matmul but bmm is efficient for batch matrix-matrix multiplication
        return torch.bmm(weights, values) # (batch_size, max_len, value_dim) (64 , 6 , 512)



# key - value (KV) attention 
"""single-query attention, where a
   single query (often from an external source) 
   attends to a sequence of keys and values."""
"""or Cross attention , 
    Example: Vision-Language Models
    In vision-language tasks like Visual Question Answering (VQA):
    Query: Encodes the question text (from a transformer-based language model).
    Key-Value Pairs: Represent image features (from a vision model like CNN or ViT).
    Cross-attention aligns the query (text) with the relevant parts of the image
    features to answer the question."""



class Attention(nn.Module):
    def __init__(self, key_dim, value_dim, device):
        super(Attention, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, keys, values, mask):
        """the Query comming from different source"""
        """Query if it was a single word representation [in the target translation for ex],
           i need to attend it to the other words in the [source] sentence."""
        
        # query won't be masked 'it's a vector not matrix'.
        # it will be used as it is. 
        # Expands query to match the sequence dimension for compatibility with keys.
        query = query.unsqueeze(1) # (batch_size, 1, key_dim)  

        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim) # (batch_size, max_len, key_dim)
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim) # (batch_size, max_len, value_dim)
        
        keys.data.masked_fill_(key_mask.bool(), 0.)
        values.data.masked_fill_(value_mask.bool(), 0.)

        max_len = keys.shape[1]

        # Computes attention weights for the single query against all tokens in keys.
        weights = torch.bmm(query, keys.permute(0,2,1)) # (batch_size, 1, max_len)
        
        weight_mask = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights)) 
        weights.data.masked_fill_(weight_mask.bool(), -float('inf'))
        
        #ignore diagonal infinity        
        weights = self.softmax(weights)
        weight_mask = torch.where(torch.isnan(weights), torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.bool(), 0.) # (batch_size, 1, max_len)
        
        return torch.bmm(weights, values).squeeze(1) # (batch_size, projected_value_dim)
    # projected_value_dim : i passed values (16,140,128) -->value_dim = 128
 

# this is the Loss or Criterion 
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    #                 pred  label
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


"""
MemexQA and MemexQA_FVTA can be trained as binary classifier "one-hot" or Multiclass classifier "select-one".
"""

# pass the structure of ur model in __init__
# pass the data in forward()
class MemexQA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        img_dim: int,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        num_label: int = 2,
        num_head: int = 2,
        num_layer: int = 1,
        mode: str = 'one-shot',
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.initialize_parameters(num_label, num_head, num_layer, mode, device)
        self.setup_image_embedding(img_dim, hidden_dim)
        self.setup_question_embedding(input_dim, key_dim, value_dim)
        self.setup_answer_embedding(input_dim, key_dim, value_dim)
        self.setup_self_attention_components(input_dim, hidden_dim, key_dim, value_dim)
        self.setup_cross_attention_components(value_dim)
        self.setup_prediction_components(key_dim, value_dim)

    def initialize_parameters(self, num_label: int, num_head: int, num_layer: int, mode: str, device: torch.device):
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer
        self.mode = mode

    def setup_image_embedding(self, img_dim: int, hidden_dim: int):
        self.img_emb = nn.Sequential(
            nn.Linear(img_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.input_layer_norm = nn.LayerNorm(hidden_dim)

    def setup_question_embedding(self, input_dim: int, key_dim: int, value_dim: int):
        self.que_emb = nn.LSTM(
            input_size=input_dim,
            hidden_size=key_dim * self.num_head // 2,
            num_layers=1,
            bidirectional=True
        )
        # self.question_projection_to_output = nn.Linear(value_dim * self.num_head, value_dim)

    def setup_answer_embedding(self, input_dim: int, key_dim: int, value_dim: int):
        hidden_size = key_dim * self.num_head // 2
        self.ans_emb = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True
        )
        # self.answer_projection_to_output = nn.Linear(value_dim * self.num_head, value_dim)

    def setup_self_attention_components(self, input_dim: int, hidden_dim: int, key_dim: int, value_dim: int):
        for i in range(self.num_head):
            # Sentence attention layers
            setattr(self, f'self_sentence_query{i+1}', nn.Linear(input_dim, key_dim))
            setattr(self, f'self_sentence_key{i+1}', nn.Linear(input_dim, key_dim))
            setattr(self, f'self_sentence_value{i+1}', nn.Linear(input_dim, value_dim))
            
            # Image attention layers
            setattr(self, f'self_image_query{i+1}', nn.Linear(hidden_dim, key_dim))
            setattr(self, f'self_image_key{i+1}', nn.Linear(hidden_dim, key_dim))
            setattr(self, f'self_image_value{i+1}', nn.Linear(hidden_dim, value_dim))

        dim = self.num_head * value_dim
        for nl in range(self.num_layer - 1):
            for i in range(self.num_head):
                setattr(self, f'self_query_{nl+1}_{i+1}', nn.Linear(dim, key_dim))
                setattr(self, f'self_key_{nl+1}_{i+1}', nn.Linear(dim, key_dim))
                setattr(self, f'self_value_{nl+1}_{i+1}', nn.Linear(dim, value_dim))

        self.self_attention = SelfAttention(key_dim, value_dim, self.device)
        self.layer_norm = nn.LayerNorm(value_dim * self.num_head)

    def setup_cross_attention_components(self, value_dim: int) -> None:
        self.key_proj = nn.Linear(value_dim * self.num_head, key_dim * self.num_head)
        self.value_proj = nn.Linear(value_dim * self.num_head, value_dim * self.num_head)
        self.attention = Attention(key_dim * self.num_head, value_dim * self.num_head, self.device)

    def setup_prediction_components(self, key_dim: int, value_dim: int):
        output_dim = {
            'one-shot': value_dim * self.num_head * 3 + key_dim * self.num_head,
            'select-one': value_dim * self.num_head * 3 + key_dim * self.num_head,
            'att-concat-one-shot': value_dim * self.num_head * 3 + key_dim * self.num_head * 2
        } 
        
        if self.mode == 'one-shot':
            self.answer_proj = nn.Linear(output_dim['one-shot'], self.num_label)
            self.criterion = LabelSmoothingLoss(2, 0.1)
        elif self.mode == 'select-one':
            self.answer_proj = nn.Linear(output_dim['select-one'], 1)
            self.criterion = LabelSmoothingLoss(4, 0.1)
        elif self.mode == 'att-concat-one-shot':
            self.answer_proj = nn.Linear(output_dim['att-concat-one-shot'], self.num_label)
            self.criterion = LabelSmoothingLoss(2, 0.1)
        else:
            raise NotImplementedError("Not implemented!")

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.2)

    # for Questions and answers
    def process_embedding(self, data, lengths, embedding_layer):
        embedded = pack_padded_sequence(
            input=data,
            lengths=lengths,
            enforce_sorted=False,
            batch_first=True
        )
        packed_output, (h, c) = embedding_layer(embedded)
        h = h.permute(1, 0, 2).contiguous()
        return h.view(h.shape[0], -1)

    def compute_self_attention(self, text, images, text_lengths, image_lengths):
        text, images = self.input_layer_norm(text), self.input_layer_norm(images)
        queries, keys, values = [], [], []
        
        for i in range(self.num_head):
            sentence_queries = getattr(self, f'self_sentence_query{i+1}')(text)
            image_queries = getattr(self, f'self_image_query{i+1}')(images)
            queries.append(torch.cat([sentence_queries, image_queries], 1))

            sentence_keys = getattr(self, f'self_sentence_key{i+1}')(text)
            image_keys = getattr(self, f'self_image_key{i+1}')(images)
            keys.append(torch.cat([sentence_keys, image_keys], 1))

            sentence_values = getattr(self, f'self_sentence_value{i+1}')(text)
            image_values = getattr(self, f'self_image_value{i+1}')(images)
            values.append(torch.cat([sentence_values, image_values], 1))

        mask = self.create_attention_mask(text.shape[1], images.shape[1], 
                                         text_lengths, image_lengths, images.shape[0])
        
        album_feat = [self.self_attention(q, k, v, mask) for q, k, v in zip(queries, keys, values)]
        album_feat = torch.cat(album_feat, -1)
        
        if self.num_layer > 1:
            album_feat = self.dropout(album_feat)
            album_feat = self.layer_norm(album_feat)
        
        return album_feat

    def create_attention_mask(self, max_text_len, max_image_len, text_lengths, image_lengths, batch_size):
        mask = np.ones((batch_size, max_text_len + max_image_len))
        for id, (tl, il) in enumerate(zip(text_lengths, image_lengths)):
            mask[id][:tl] = 0
            mask[id][max_text_len:max_text_len + il] = 0
        return torch.Tensor(mask).to(self.device)


    
    def forward(self, 
                question,
                question_lengths,
                answer, 
                answer_lengths, 
                text,
                text_lengths, 
                images, 
                image_lengths, 
                label):
                    
        assert question.shape[1] == max(question_lengths), "Mismatch between question and max sequence length!"

        question_embed = self.process_embedding(question, question_lengths, self.que_emb)
        answer_embed = self.process_embedding(answer, answer_lengths, self.ans_emb)
        images = self.img_emb(images)
        
        album_feat = self.compute_self_attention(text, images, text_lengths, image_lengths)
        album_keys = self.key_proj(album_feat)
        album_values = self.value_proj(album_feat)
        
        atts_question = self.attention(question_embed,
                                       album_keys, 
                                       album_values, 
                                       self.create_attention_mask(text.shape[1],
                                                                   images.shape[1],
                                                                   text_lengths, 
                                                                   image_lengths, 
                                                                   images.shape[0]))

        if self.mode == 'one-shot':
            outputs = torch.cat([question_embed, answer_embed, atts_question, 
                               answer_embed * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs)[:, 0]
            loss = self.criterion(outputs, label)
        
        elif self.mode == 'select-one':
            outputs = torch.cat([question_embed, answer_embed, atts_question, 
                               answer_embed * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs.view(-1, 4))
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        
        elif self.mode == 'att-concat-one-shot':
            atts_answer = self.attention(answer_embed, album_keys, album_values, 
                                       self.create_attention_mask(text.shape[1], images.shape[1],
                                                                 text_lengths, image_lengths, images.shape[0]))
            outputs = torch.cat([question_embed, answer_embed, atts_answer, 
                               atts_question, atts_question * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs)[:, 0]
            loss = self.criterion(outputs, label)
        
        else:
            raise NotImplementedError("Not implemented!")

        return prediction, loss


# FVT Attention --> as replacement for Cross-attention
class FVTA(nn.Module):
    def __init__(self, key_dim, value_dim, device):
        super(FVTA, self).__init__()
        self.device=device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, query_lengths, keys, values, mask):
        # query = (batch_size, seq_len, key_dim)
        # query_lengths = (batch_size)

        # create a simple mask for (16,12,128) and text lengths = len(16)
        batch_size, max_query_len = query.shape[0], query.shape[1]
        mask_query = np.ones((batch_size, max_query_len))
        for id, ql in enumerate(query_lengths):
            mask_query[id][:ql] = np.zeros_like(mask_query[id][:ql])
        mask_query = torch.Tensor(mask_query).to(self.device)

        mask_query = mask_query.unsqueeze(-1).repeat(1,1,self.key_dim) #(16,12,32)
        query.data.masked_fill_(mask_query.bool(), 0.) # (batch_size, seq_len, key_dim)

        # keys   = (batch_size, max_len, key_dim)
        # values = (batch_size, max_len, value_dim)
        key_mask = mask.unsqueeze(-1).repeat(1,1,self.key_dim) # (batch_size, max_len, key_dim)
        value_mask = mask.unsqueeze(-1).repeat(1,1,self.value_dim) # (batch_size, max_len, value_dim)
        
        keys.data.masked_fill_(key_mask.bool(), 0.)
        values.data.masked_fill_(value_mask.bool(), 0.)

        max_len = keys.shape[1]
        weights = torch.bmm(query, keys.permute(0,2,1)) # (batch_size, seq_len, max_len)
        weight_mask = torch.where(weights == 0., torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.bool(), -float('inf'))

        weights = self.softmax(weights)
        weight_mask = torch.where(torch.isnan(weights), torch.ones_like(weights), torch.zeros_like(weights))
        weights.data.masked_fill_(weight_mask.bool(), 0.) # (batch_size, seq_len, max_len)
        
        # Mean Pooling 
        # (16,12,140)*(16,140,128)-->(16,12,128)-->mean(dim=1)-->(16,128)
        return torch.mean(torch.bmm(weights, values), dim = 1) # (batch_size, projected_value_dim)


"""
Story So Far:
You start with a question tensor of shape (8, 12, 768), where e.g.:
    8 represents the batch size (number of questions),
    12 is the sequence length (tokens per question),
    768 is the feature dimension.

Step 1: Packing the Sequence
    To efficiently process variable-length sequences, 
    we pack the input using pack_padded_sequence, 
    incorporating the actual lengths of each question. 
    After packing, the tensor is reshaped from (8 * 12, 768) → (96, 768).
    However, since packing removes padding, the effective 
    size reduces to (32, 768)—where 32 represents the 
    total valid tokens in the batch. The remaining 64 tokens
    (96 - 32) were just padding and are now ignored.

Step 2: Passing Through LSTM
    The packed sequence (32, 768) is then fed into the LSTM, which generates:
    Hidden states (h, c) of shape (8, 128), representing the processed batch-wise 
    context.
    Packed output (packed_question) of shape (32, 128), where the LSTM models 
    the actual sequence without padding.

Step 3: Unpacking the Sequence
    We use pad_packed_sequence to transform (32, 128) back into a structured 
    format:
    (8, 12, 128) → (batch_size, max_seq_len, hidden_size).
    This is now a refined representation of the input, enriched with sequential context from the LSTM.

Final Step: Feeding into FVT Attention
    Instead of using the original raw input (8, 12, 768), 
    we now pass the LSTM-processed output (8, 12, 128) 
    into the FVT attention mechanism, ensuring that the model
    attends to a contextually enriched representation rather
    than the unprocessed embeddings.
"""

class MemexQA_FVTA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        img_dim: int,
        hidden_dim: int,
        key_dim: int,
        value_dim: int,
        num_label: int = 2,
        num_head: int = 2,
        num_layer: int = 1,
        mode: str = 'one-shot',
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.initialize_parameters(num_label, num_head, num_layer, mode, device)
        self.setup_embeddings(input_dim, img_dim, hidden_dim, key_dim, value_dim)
        self.setup_self_attention(input_dim, hidden_dim, key_dim, value_dim)
        self.setup_fvt_attention(value_dim)
        self.setup_prediction_layers(key_dim, value_dim)

    def initialize_parameters(self, num_label: int, num_head: int, num_layer: int, mode: str, device: torch.device):
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer
        self.mode = mode

    def setup_embeddings(self, input_dim: int, img_dim: int, hidden_dim: int, key_dim: int, value_dim: int):
        self.img_emb = nn.Sequential(
            nn.Linear(img_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.input_layer_norm = nn.LayerNorm(hidden_dim)
        
        self.que_emb = nn.LSTM(input_size=input_dim, 
                             hidden_size=key_dim * self.num_head // 2, 
                             num_layers=1, 
                             bidirectional=True)
        
        ans_hidden_size = key_dim * self.num_head // 2 
        self.ans_emb = nn.LSTM(input_size=input_dim, 
                             hidden_size=ans_hidden_size, 
                             num_layers=1, 
                             bidirectional=True)

    def setup_self_attention(self, input_dim: int, hidden_dim: int, key_dim: int, value_dim: int):
        for i in range(self.num_head):
            setattr(self, f'self_sentence_query{i+1}', nn.Linear(input_dim, key_dim))
            setattr(self, f'self_sentence_key{i+1}', nn.Linear(input_dim, key_dim))
            setattr(self, f'self_sentence_value{i+1}', nn.Linear(input_dim, value_dim))
            setattr(self, f'self_image_query{i+1}', nn.Linear(hidden_dim, key_dim))
            setattr(self, f'self_image_key{i+1}', nn.Linear(hidden_dim, key_dim))
            setattr(self, f'self_image_value{i+1}', nn.Linear(hidden_dim, value_dim))

        dim = self.num_head * value_dim
        for nl in range(self.num_layer - 1):
            for i in range(self.num_head):
                setattr(self, f'self_query_{nl+1}_{i+1}', nn.Linear(dim, key_dim))
                setattr(self, f'self_key_{nl+1}_{i+1}', nn.Linear(dim, key_dim))
                setattr(self, f'self_value_{nl+1}_{i+1}', nn.Linear(dim, value_dim))

        self.self_attention = SelfAttention(key_dim, value_dim, self.device)
        self.layer_norm = nn.LayerNorm(value_dim * self.num_head)
        self.dropout = nn.Dropout(p=0.2)

    def setup_fvt_attention(self, value_dim: int) -> None:
        self.key_proj = nn.Linear(value_dim * self.num_head, key_dim * self.num_head)
        self.value_proj = nn.Linear(value_dim * self.num_head, value_dim * self.num_head)
        self.attention = FVTA(key_dim * self.num_head, value_dim * self.num_head, self.device)

    def setup_prediction_layers(self, key_dim: int, value_dim: int) -> None:
        input_dim = {
            'one-shot': value_dim * self.num_head * 3 + key_dim * self.num_head,
            'select-one': value_dim * self.num_head * 3 + key_dim * self.num_head,
            'att-concat-one-shot': value_dim * self.num_head * 3 + key_dim * self.num_head * 2
        }[self.mode]
        
        output_dim = 1 if self.mode == 'select-one' else self.num_label
        self.answer_proj = nn.Linear(input_dim, output_dim)
        
        self.criterion = LabelSmoothingLoss(2 if 'one-shot' in self.mode else 4, 0.1)
        self.softmax = nn.Softmax(dim=-1)

    def process_sequence(self, sequence: torch.Tensor, lengths: torch.Tensor, embedding: nn.Module):
        packed = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h, c) = embedding(packed)
        fatt, fatt_lengths = pad_packed_sequence(packed_output)
        fatt = fatt.permute(1, 0, 2)
        h = h.permute(1, 0, 2).contiguous()
        return fatt, fatt_lengths, h.view(h.shape[0], -1)

    def compute_self_attention(self, text: torch.Tensor, images: torch.Tensor, 
                              text_lengths: torch.Tensor, image_lengths: torch.Tensor):
        text, images = self.input_layer_norm(text), self.input_layer_norm(images)
        queries, keys, values = [], [], []
        
        for i in range(self.num_head):
            queries.append(torch.cat([
                getattr(self, f'self_sentence_query{i+1}')(text),
                getattr(self, f'self_image_query{i+1}')(images)
            ], 1))
            keys.append(torch.cat([
                getattr(self, f'self_sentence_key{i+1}')(text),
                getattr(self, f'self_image_key{i+1}')(images)
            ], 1))
            values.append(torch.cat([
                getattr(self, f'self_sentence_value{i+1}')(text),
                getattr(self, f'self_image_value{i+1}')(images)
            ], 1))

        mask = self._create_mask(text.shape[1], images.shape[1], text_lengths, image_lengths, text.shape[0])
        album_feat = torch.cat([self.self_attention(q, k, v, mask) for q, k, v in zip(queries, keys, values)], -1)
        
        if self.num_layer > 1:
            album_feat = self.dropout(album_feat)
            album_feat = self.layer_norm(album_feat)
            
            for nl in range(self.num_layer - 1):
                album_feat = self._additional_self_attention(album_feat, mask, nl)
        
        return album_feat

    def create_mask(self, max_text_len: int, max_image_len: int, 
                    text_lengths: torch.Tensor, image_lengths: torch.Tensor, 
                    batch_size: int) :
        mask = np.ones((batch_size, max_text_len + max_image_len))
        for idx, (tl, il) in enumerate(zip(text_lengths, image_lengths)):
            mask[idx][:tl] = 0
            mask[idx][max_text_len:max_text_len + il] = 0
        return torch.Tensor(mask).to(self.device)

    def additional_self_attention(self, album_feat: torch.Tensor, mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        queries, keys, values = [], [], []
        for i in range(self.num_head):
            queries.append(getattr(self, f'self_query_{layer_idx+1}_{i+1}')(album_feat))
            keys.append(getattr(self, f'self_key_{layer_idx+1}_{i+1}')(album_feat))
            values.append(getattr(self, f'self_value_{layer_idx+1}_{i+1}')(album_feat))
        
        new_album_feat = torch.cat([self.self_attention(q, k, v, mask) 
                                  for q, k, v in zip(queries, keys, values)], -1)
        
        if layer_idx < self.num_layer - 2:
            new_album_feat = self.dropout(new_album_feat)
            new_album_feat = self.layer_norm(new_album_feat)
        
        return new_album_feat

    def forward(self, question, question_lengths, answer, answer_lengths, text,
                text_lengths, images, image_lengths, label):
        question_fatt, question_fatt_lengths, question = self.process_sequence(question, question_lengths, self.que_emb)
        answer_fatt, answer_fatt_lengths, answer = self.process_sequence(answer, answer_lengths, self.ans_emb)
        images = self.img_emb(images)
        
        album_feat = self._compute_self_attention(text, images, text_lengths, image_lengths)
        album_keys = self.key_proj(album_feat)
        album_values = self.value_proj(album_feat)
        
        mask = self._create_mask(text.shape[1], images.shape[1], text_lengths, image_lengths, question.shape[0])
        atts_question = self.attention(question_fatt, question_fatt_lengths, album_keys, album_values, mask)

        if self.mode == 'one-shot':
            outputs = torch.cat([question, answer, atts_question, answer * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs)[:, 0]
            loss = self.criterion(outputs, label)
        elif self.mode == 'select-one':
            outputs = torch.cat([question, answer, atts_question, answer * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs.view(-1, 4))
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        elif self.mode == 'att-concat-one-shot':
            atts_answer = self.attention(answer_fatt, answer_lengths, album_keys, album_values, mask)
            outputs = torch.cat([question, answer, atts_answer, atts_question, atts_question * atts_question], 1)
            outputs = self.answer_proj(outputs)
            prediction = self.softmax(outputs)[:, 0]
            loss = self.criterion(outputs, label)
        else:
            raise NotImplementedError("Not implemented!")

        return prediction, loss
