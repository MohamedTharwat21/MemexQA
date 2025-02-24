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



# pass the structure of ur model in __init__
# pass the data in forward()
class MemexQA(nn.Module):
    def __init__(self, input_dim, img_dim, hidden_dim, key_dim, value_dim, num_label=2, num_head=2, num_layer=1, mode='one-shot', device=torch.device("cpu")):
        super(MemexQA, self).__init__()      
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer #for lstm
        self.mode = mode
        
        # img_emb
        # self.img_emb = nn.Linear(img_dim, hidden_dim) #in one jump
        self.img_emb = nn.Sequential(
            nn.Linear(img_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        self.input_layer_norm = nn.LayerNorm(hidden_dim)

        # que_emb
        """
        Without dividing by 2, the output would have 
        dimensions 2 * key_dim * self.num_head, which 
        might not match the expected dimensions in 
        subsequent layers of the model."""

        # The hidden size of key_dim * self.num_head // 2 
        # is chosen to ensure that the final output dimension
        # of the bidirectional LSTM matches the required dimensions 
        # (key_dim * num_head).
        
        # effective hidden size=64×2=128
        self.que_emb = nn.LSTM(input_size=input_dim, hidden_size=key_dim*self.num_head//2, num_layers=1, bidirectional=True)
        self.question_projection_to_ouput = nn.Linear(value_dim*self.num_head , value_dim) # nn.Linear(128 , 32)
        
        # ans_emb
        if self.mode == 'att-concat-one-shot':
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=key_dim*self.num_head//2, num_layers=1, bidirectional=True)
        else:
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=value_dim*self.num_head//2, num_layers=1, bidirectional=True)
        self.answer_projection_to_ouput = nn.Linear(value_dim*self.num_head , value_dim) # nn.Linear(128 , 32)        

        # self attention
        # Self-attention for sentence and image which is the context or the album infor
        for i in range(self.num_head):
            # focus the difference between self like this and self for self attention
            setattr(self, 'self_sentence_query'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_key'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_value'+str(i+1), nn.Linear(input_dim, value_dim))
            
            # we projected the image to the hidden dim
            # so start from it 
            setattr(self, 'self_image_query'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_key'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_value'+str(i+1), nn.Linear(hidden_dim, value_dim))
        
        modern_implemenation = False
        if modern_implemenation :
            # Self-attention for sentence and image
            """modern implementations of multi-head attention."""
            # self.sentence_query = nn.ModuleList([nn.Linear(input_dim, key_dim) for _ in range(self.num_head)])
            # self.sentence_key = nn.ModuleList([nn.Linear(input_dim, key_dim) for _ in range(self.num_head)])
            # self.sentence_value = nn.ModuleList([nn.Linear(input_dim, value_dim) for _ in range(self.num_head)])
            
            # self.image_query = nn.ModuleList([nn.Linear(hidden_dim, key_dim) for _ in range(self.num_head)])
            # self.image_key = nn.ModuleList([nn.Linear(hidden_dim, key_dim) for _ in range(self.num_head)])
            # self.image_value = nn.ModuleList([nn.Linear(hidden_dim, value_dim) for _ in range(self.num_head)])
            pass

        # Shared components
        self.self_attention = SelfAttention(key_dim, value_dim, self.device)
        self.layer_norm = nn.LayerNorm(value_dim*self.num_head)#32*4=128

        # More layers :) this in case i make in the forward operation many layers
        # this is for album_feat below which ends with 128 ,(bs*4 ,...,128).(128,32)
        dim = self.num_head * value_dim    #128
        for nl in range(self.num_layer-1):
            for i in range(self.num_head): # 4 times
                setattr(self, f'self_query_{nl+1}_{i+1}' , nn.Linear(dim, key_dim))
                setattr(self, f'self_key_{nl+1}_{i+1}' , nn.Linear(dim, key_dim))
                setattr(self, f'self_value_{nl+1}_{i+1}' , nn.Linear(dim, value_dim))
        
        # question attention
        # here Cross Attention 
        """as the question will be as a Query to be passed to KV Attention ,
           so the key_proj and value_proj will be Queried by the Question """
        
        # (128,128)
        """the projection here is for the Key Value matrices themselves 
           to be learned for the Question attention ."""
        self.key_proj   = nn.Linear(value_dim*self.num_head, key_dim*self.num_head)
        self.value_proj = nn.Linear(value_dim*self.num_head, value_dim*self.num_head)        
        self.attention  = Attention(key_dim*self.num_head, value_dim*self.num_head, self.device)

        # Prediction
        if self.mode == 'one-shot':
            # (128*4 , 2)
            self.answer_proj = nn.Linear(value_dim*self.num_head*3 + key_dim*self.num_head, self.num_label)
        elif self.mode == 'select-one': # NOT GREAT
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head, 1)
        elif self.mode == 'att-concat-one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head*2, self.num_label)
        else:
             raise NotImplementedError("Not implemented!")

        # criterion
        # self.criterion = nn.CrossEntropyLoss()
        if 'one-shot' in self.mode:
            self.criterion = LabelSmoothingLoss(2, 0.1)
        elif 'select-one' in self.mode:
            self.criterion = LabelSmoothingLoss(4, 0.1)
               
        self.softmax = nn.Softmax(dim=-1)
        # Additional Techniques
        self.dropout = nn.Dropout(p=0.2)


    
    # def pad_text_image(self , text , image):
    #     # take a shallow copy of the image 
    #     image_cp = image.unsqueeze(1) 

    #     # Determine max sequence length (dim=1) and max feature width (dim=2)
    #     max_seq_length = max(text.shape[1], image_cp.shape[1])
    #     max_feature_width = max(text.shape[2], image_cp.shape[2])

    #     # Pad sequences along dimensions 1 and 2
    #     padded_text = F.pad(text, (0, 0, 0, max_feature_width - text.shape[2], 0, max_seq_length - text.shape[1]))
    #     padded_image = F.pad(image_cp, (0, 0, 0, max_feature_width - image_cp.shape[2], 0, max_seq_length - image_cp.shape[1]))

    #     # Concatenate along the sequence length dimension (dim=1)
    #     # dim = 1 --> to concatenate max_image_len and max_text_len
    #     cat = torch.cat([padded_text, padded_image], dim=1)
 
    #     return cat # (16, max_seq_length, max_feature_width, 32)
    

    def forward(self, question, question_lengths, answer, answer_lengths, text,\
                text_lengths, images, image_lengths, label):
        # """
        # why i used pack_padded_sequence while i used packing in collate_fn?
        # Standard LSTMs process every time step in the sequence, including padded positions (i.e., zeros).
        # Using pack_padded_sequence ensures the LSTM only processes non-padded tokens, skipping unnecessary computations.
        # """
        assert question.shape[1] == max(question_lengths), "Mismatch between question and max sequence length!"  
        # question_lengths, perm_idx = question_lengths.sort(descending=True)
        # question = question[perm_idx]  # Sort question tensor to match

        # instead of permute use batch_first = True in pack_padded_sequence
        # enforce_sorted=True needs the lengths to be descended sorted 
        # batch_first here is essential 
        question_embed = pack_padded_sequence(input=question,
                                              lengths=question_lengths,
                                              enforce_sorted=False,
                                              batch_first=True)        
        packed_question, (h,c) = self.que_emb(question_embed)
        h = h.permute(1,0,2).contiguous() #(batch, num_layer, hidden_size) #hidden_size=key_dim*self.num_head//2
        # h.shape=(16,2,64)
        question = h.view(h.shape[0], -1) #(batch, num_layer*hidden_size) (bs,128)
        # question = h.shape = (16,2*64)

        # answer
        answer_embed = pack_padded_sequence(input=answer, 
                                            lengths=answer_lengths,
                                            enforce_sorted=False,
                                            batch_first=True)       
        packed_answer, (h,c) = self.ans_emb(answer_embed)
        h = h.permute(1,0,2).contiguous()
        # h.shape=(16,2,64)
        answer = h.view(h.shape[0], -1) # (bs,128)
        # answer = h.shape = (16,2*64)
        
        # images and text
        #(4*8,40,768)
        images  = self.img_emb(images) # (bs*4, max_image_length, hidden_dim) #(4*8,40,768)
        batch_size_times_4 = question.shape[0] #batch first
        # context 
        text, images = self.input_layer_norm(text), self.input_layer_norm(images)

        # """
        # up to now :
        #    we have text , images , question , answer Embedding
        #    this is the step of [Context Encoder] 
        #    [4] used LSTM to encode all the contextual information
        #    but , I will replace it using the Self-Attention
        #    the input for SA is : are image/text embedding,
        #    [this is the context , not the QAs]
        # """

        # these 3 matrices as we were doing in the transformers
        # we have a context (album_info) which i need to make it Self-Attended
        # simply make 3 copies of the input[images/text]
        queries, keys, values = [], [], []
        for i in range(self.num_head):
            # first copy

            """
            # Retrieve the dynamic layer and apply it
            layer = getattr(self, 'self_sentence_query' + str(i+1))
            outputs.append(layer(text)) """
            
            # sentence_quries_layer = getattr(self,'self_sentence_query'+str(i+1))
            # sentence_quries_layer(text)
            sentence_quries = getattr(self,'self_sentence_query'+str(i+1))(text)
            image_quries    = getattr(self,'self_image_query'+str(i+1))(images)   
            # queries.append(self.pad_text_image(sentence_quries , image_quries))
            queries.append(torch.cat([sentence_quries, image_quries], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            # second copy
            sentence_keys = getattr(self,'self_sentence_key'+str(i+1))(text)
            image_keys    = getattr(self,'self_image_key'+str(i+1))(images)  
            # keys.append(self.pad_text_image(sentence_keys , image_keys)) # (num_head, bs*4, text_len+image_len, key_dim)
            keys.append(torch.cat([sentence_keys, image_keys], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            # third copy
            sentence_values = getattr(self,'self_sentence_value'+str(i+1))(text)
            image_values    = getattr(self,'self_image_value'+str(i+1))(images)         
            # values.append(self.pad_text_image(sentence_values , image_values)) # (num_head, bs*4, text_len+image_len, value_dim)
            values.append(torch.cat([sentence_values, image_values], 1)) # (num_head, bs*4, text_len+image_len, value_dim)
       
        # mask for self_attention
        max_text_len, max_image_len = text.shape[1], images.shape[1]
        mask = np.ones((batch_size_times_4 , max_text_len+max_image_len)) #batch_size=bs*4
        for id, (tl, il) in enumerate(zip(text_lengths, image_lengths)):
            mask[id][:tl] = np.zeros_like(mask[id][:tl])
            mask[id][max_text_len:max_text_len+il] = np.zeros_like(mask[id][max_text_len:max_text_len+il])          
            # mask[id][:tl] = 0  # Unmask valid sequences
            # mask[id][max_text_len:max_text_len+il] = 0  # Unmask valid images
        mask = torch.Tensor(mask).to(self.device)
        
        word_level_masking = False
        if word_level_masking :
            ...
            # if you need word level masking ,but i did not pass the actual word lengths 
            # Step 2: Create word-level mask
            # word_mask = (text.abs().sum(dim=-1) != 0).float()  # (bs, max_seq_len, max_word_len)
            # # Step 3: Combine both masks (if needed)
            # # This will ensure both sequence-level and word-level masking
            # final_mask = mask.unsqueeze(-1) * word_mask  # (bs, max_seq_len, max_word_len)
     
        # album_feat that will contain the encoded and well presented album contents
        # the context here is the album data ,remember :)
        album_feat = [] #(bs*4 , max_text_len+max_image_len , value_dim*num_head)        
        """queries, keys, values = [], [], [] ; are lists each one of 4 items , each item repre just one head."""
        for i in range(self.num_head): #4 times
            """these are the keys , queries and values of the context (images/texts)"""
            """pass the 1st key,query,value to make the self attention""" 
            # print(f'queries[{i}].shape {queries[i].shape} ,keys[{i}].shape {keys[i].shape}, values[{i}].shape {values[i].shape}')
            self_att = self.self_attention(queries[i], keys[i], values[i], mask)
            album_feat.append(self_att) 
                                               # it ends with 128
        album_feat = torch.cat(album_feat, -1) # (bs*4, ..., value_dim*num_head)
        
        if self.num_layer > 1: #in case of overfitting due to the increased number of the layers
            album_feat = self.dropout(album_feat)
            album_feat = self.layer_norm(album_feat)
        
        """the context data (album_feat) will be also self-attented again with the num_layers
           here the condition will not be achieved as i used self.num_layers = 1
           ,so ignore it for now and go for the next ."""
        
        # More self-attention-layers
        for nl in range(self.num_layer-1):
            queries, keys, values = [], [], []
            for i in range(self.num_head):
                query = getattr(self, f'self_query_{nl+1}_{i+1}')(album_feat)
                queries.append(query)

                key = getattr(self, f'self_key_{nl+1}_{i+1}')(album_feat)
                keys.append(key)

                value = getattr(self, f'self_value_{nl+1}_{i+1}')(album_feat)
                values.append(value)

            new_album_feat = []
            for i in range(self.num_head):
                self_att = self.self_attention(queries[i], keys[i], values[i], mask)
                new_album_feat.append(self_att)
            
            new_album_feat = torch.cat(new_album_feat, -1) # (bs*4, text_len+image_len, value_dim*num_head)

            if nl < self.num_layer-2:
                album_feat = self.dropout(album_feat)
                album_feat = self.layer_norm(album_feat)

            album_feat = new_album_feat
      
        # Cross-Attention [the Question is the Query and the Context Encoder
        # is the Key and Value ]


        # Attention
        album_keys   = self.key_proj(album_feat) # (bs*4, ..., key_dim)
        album_values = self.value_proj(album_feat) # (bs*4, ..., value_dim)
        # query will be the question
        """here is making the attention between the context(images/text)and the question
           ,the KV attention is that key and value comes from the same source ,
            here is the album or the context.
            And the Query comes from another source which is the training data
            which is the Question embdedding  """
        # Projection
                                    #   query     keys        values
        atts_question = self.attention(question, album_keys, album_values, mask) # (bs*4, value_dim)       
        """
        torch.cat → Combines four sources of information:
            question → Context from the question (semantic meaning).
            answer → Proposed answer encoding.
            atts_question → Attention weights related to the question.
            interaction (answer * atts_question) → How answer interacts with attention.
            This allows self.answer_proj (a fully connected layer) to process 
            the combined information before making a classification.
        """
        if self.mode == 'one-shot':
            # question = self.question_projection_to_ouput(question) #(bs*4,128)*(128,32)-->(bs*4,32)
            # answer = self.answer_projection_to_ouput(answer) #(bs*4,128)*(128,32)-->(bs*4,32)           
            outputs = torch.cat([question,  #(bs*4,128)
                                  answer,   #(bs*4,128)
                                  atts_question,   #(bs*4,128)
                                  # element wise multiplication
                                  #(bs*4,128).(bs*4,128)-->(bs*4,128)
                                  answer*atts_question],1) # (bs*4, value_dim*self.num_head*3 + key_dim*self.num_head)
                                                           # (16,4*128)
            # (16,4*128).(4*128,2) -->(16,2)
            # A linear layer is applied to transform the concatenated feature vector (bs*4, 4*128)
            # into a 2D output (binary classification).
            # The final shape after projection is (bs*4, 2),
            # where each row has two logits (scores) for two classes (e.g., class 0 and class 1).
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            # This extracts the probability of the first class (class 0).
            """what is the probability of the given sample to be Zero(means right ans)."""
            prediction = self.softmax(outputs)[:, 0] # (bs*4) 
            loss = self.criterion(outputs, label)      
        elif self.mode == 'select-one':
            outputs = torch.cat([question, answer, atts_question, answer*atts_question], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 1)
            prediction = self.softmax(outputs.view(-1, 4)) # (bs, 4)
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        elif self.mode == 'att-concat-one-shot':
            atts_answer = self.attention(answer, album_keys, album_values, mask) # (bs*4, value_dim)
            outputs = torch.cat([question, answer, atts_answer, atts_question, atts_question*atts_question], 1) # (bs*4, value_dim*3+key_dim*2)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
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


class MemexQA_FVTA(nn.Module):
    def __init__(self, input_dim, img_dim, hidden_dim, key_dim, value_dim, num_label=2, num_head=2, num_layer=1, mode='one-shot', device=torch.device("cpu")):
        super(MemexQA_FVTA, self).__init__()
        self.device = device
        self.num_label = num_label
        self.num_head = num_head
        self.num_layer = num_layer
        self.mode = mode
        
        # img_emb
        # self.img_emb = nn.Linear(img_dim, hidden_dim)
        self.img_emb = nn.Sequential(
            nn.Linear(img_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        self.input_layer_norm = nn.LayerNorm(hidden_dim)

        # que_emb
        self.que_emb = nn.LSTM(input_size=input_dim, hidden_size= key_dim*self.num_head//2, num_layers=1, bidirectional=True)

        # ans_emb
        if self.mode == 'att-concat-one-shot':
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=key_dim*self.num_head//2, num_layers=1, bidirectional=True)
        else:
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=value_dim*self.num_head//2, num_layers=1, bidirectional=True)

        # self attention
        for i in range(self.num_head):
            setattr(self, 'self_sentence_query'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_key'+str(i+1), nn.Linear(input_dim, key_dim))
            setattr(self, 'self_sentence_value'+str(i+1), nn.Linear(input_dim, value_dim))
            setattr(self, 'self_image_query'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_key'+str(i+1), nn.Linear(hidden_dim, key_dim))
            setattr(self, 'self_image_value'+str(i+1), nn.Linear(hidden_dim, value_dim))
        
        self.self_attention = SelfAttention(key_dim, value_dim, self.device)
        self.layer_norm = nn.LayerNorm(value_dim*self.num_head)

        # More layers
        dim = self.num_head * value_dim
        for nl in range(self.num_layer-1):
            for i in range(self.num_head):
                setattr(self, 'self_query_{}_{}'.format(nl+1, i+1), nn.Linear(dim, key_dim))
                setattr(self, 'self_key_{}_{}'.format(nl+1, i+1), nn.Linear(dim, key_dim))
                setattr(self, 'self_value_{}_{}'.format(nl+1, i+1), nn.Linear(dim, value_dim))

        # question attention
        self.key_proj   = nn.Linear(value_dim*self.num_head, key_dim*self.num_head)
        self.value_proj = nn.Linear(value_dim*self.num_head, value_dim*self.num_head)
        """the first difference"""
        """FVT attention will be used instead the normal KV attention"""
        self.attention  = FVTA(key_dim*self.num_head, value_dim*self.num_head, self.device)

        # Prediction
        if self.mode == 'one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head, self.num_label)
        elif self.mode == 'select-one': # NOT GREAT
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head, 1)
        elif self.mode == 'att-concat-one-shot':
            self.answer_proj = nn.Linear(value_dim*self.num_head*3+key_dim*self.num_head*2, self.num_label)
        else:
             raise NotImplementedError("Not implemented!")

        # criterion
        # self.criterion = nn.CrossEntropyLoss()
        if 'one-shot' in self.mode:
            self.criterion = LabelSmoothingLoss(2, 0.1)
        elif 'select-one' in self.mode:
            self.criterion = LabelSmoothingLoss(4, 0.1)
        self.softmax = nn.Softmax(dim=-1)

        # Additional Techniques
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label):
        # pack_padded_sequence
        # Converts a padded batch of variable-length sequences into a packed format for efficient processing in RNNs.
        # Used before feeding sequences into an RNN.

        # pad_packed_sequence
        # Converts a packed sequence back into a padded sequence (after processing by an RNN.)
        # Used after an RNN to recover the original batch format.
                
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
        
        # Question
        """the second difference is here.
           (question_fatt, question_lengths) --> this will be passed to the FTA attention """
        question_embed = pack_padded_sequence(input=question, 
                                              lengths=question_lengths,
                                              enforce_sorted=False,
                                              batch_first=True)
        
        packed_question, (h,c) = self.que_emb(question_embed)
        question_fatt, question_lengths = pad_packed_sequence(packed_question)
        question_fatt = question_fatt.permute(1,0,2)
        h = h.permute(1,0,2).contiguous()
        question = h.view(h.shape[0], -1)
        
        # Answer
        answer_embed = pack_padded_sequence(input=answer,
                                            lengths=answer_lengths,
                                            enforce_sorted=False,
                                            batch_first=True)
        
        packed_answer, (h,c) = self.ans_emb(answer_embed)
        answer_fatt, answer_lengths = pad_packed_sequence(packed_answer)
        answer_fatt = answer_fatt.permute(1,0,2)
        h = h.permute(1,0,2).contiguous()
        answer = h.view(h.shape[0], -1)
        
        images   = self.img_emb(images) # (bs*4, ..., hidden_dim)

        batch_size = question.shape[0]
        text, images = self.input_layer_norm(text), self.input_layer_norm(images)

        queries, keys, values = [], [], []
        for i in range(self.num_head):
            sentence_quries = getattr(self,'self_sentence_query'+str(i+1))(text)
            image_quries    = getattr(self,'self_image_query'+str(i+1))(images)
            queries.append(torch.cat([sentence_quries, image_quries], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            sentence_keys = getattr(self,'self_sentence_key'+str(i+1))(text)
            image_keys    = getattr(self,'self_image_key'+str(i+1))(images)
            keys.append(torch.cat([sentence_keys, image_keys], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            sentence_values = getattr(self,'self_sentence_value'+str(i+1))(text)
            image_values    = getattr(self,'self_image_value'+str(i+1))(images)
            values.append(torch.cat([sentence_values, image_values], 1)) # (num_head, bs*4, text_len+image_len, value_dim)
        
        

        # mask for self_attention
        max_text_len, max_image_len = text.shape[1], images.shape[1]
        mask = np.ones((batch_size, max_text_len+max_image_len))
        for id, (tl, il) in enumerate(zip(text_lengths, image_lengths)):
            mask[id][:tl] = np.zeros_like(mask[id][:tl])
            mask[id][max_text_len:max_text_len+il] = np.zeros_like(mask[id][max_text_len:max_text_len+il])
        mask = torch.Tensor(mask).to(self.device)

        album_feat = []
        for i in range(self.num_head):
            self_att = self.self_attention(queries[i], keys[i], values[i], mask)
            album_feat.append(self_att)
        album_feat = torch.cat(album_feat, -1) # (bs*4, ..., value_dim*num_head)
        if self.num_layer > 1:
            album_feat = self.dropout(album_feat)
            album_feat = self.layer_norm(album_feat)

        # More self-attention-layers
        for nl in range(self.num_layer-1):
            queries, keys, values = [], [], []
            for i in range(self.num_head):
                query = getattr(self, 'self_query_{}_{}'.format(nl+1, i+1))(album_feat)
                queries.append(query)

                key = getattr(self, 'self_key_{}_{}'.format(nl+1, i+1))(album_feat)
                keys.append(key)

                value = getattr(self, 'self_value_{}_{}'.format(nl+1, i+1))(album_feat)
                values.append(value)

            new_album_feat = []
            for i in range(self.num_head):
                self_att = self.self_attention(queries[i], keys[i], values[i], mask)
                new_album_feat.append(self_att)
            new_album_feat = torch.cat(new_album_feat, -1) # (bs*4, text_len+image_len, value_dim*num_head)

            if nl < self.num_layer-2:
                album_feat = self.dropout(album_feat)
                album_feat = self.layer_norm(album_feat)

            album_feat = new_album_feat

        # Attention
        album_keys   = self.key_proj(album_feat) # (bs*4, ..., key_dim)
        album_values = self.value_proj(album_feat) # (bs*4, ..., value_dim)

        # Projection
        # this is FVT attention , not cross-attention attention
        atts_question = self.attention(question_fatt, question_lengths, album_keys, album_values, mask) # (bs*4, value_dim)
        
        
        if self.mode == 'one-shot':
            outputs = torch.cat([question, answer, atts_question, answer*atts_question], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
            loss = self.criterion(outputs, label)
        elif self.mode == 'select-one':
            outputs = torch.cat([question, answer, atts_question, answer*atts_question], 1) # (bs*4, value_dim*3+key_dim)
            outputs = self.answer_proj(outputs) # (bs*4, 1)
            prediction = self.softmax(outputs.view(-1, 4)) # (bs, 4)
            loss = self.criterion(prediction, torch.LongTensor(np.zeros(prediction.shape[0])).to(self.device))
        elif self.mode == 'att-concat-one-shot':
            atts_answer = self.attention(answer_fatt, album_keys, album_values, mask) # (bs*4, value_dim)
            outputs = torch.cat([question, answer, atts_answer, atts_question, atts_question*atts_question], 1) # (bs*4, value_dim*3+key_dim*2)
            outputs = self.answer_proj(outputs) # (bs*4, 2)
            prediction = self.softmax(outputs)[:, 0] # (bs*4)
            loss = self.criterion(outputs, label)
        else:
            raise NotImplementedError("Not implemented!")

        return prediction, loss