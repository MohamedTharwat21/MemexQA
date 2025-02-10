from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as ag
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



from cross_attention import Attention
from self_attention import SelfAttention


# this is the Loss or Criterion 
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


"""
model = MemexQA(input_dim=768,  #word embedding of bert
             img_dim=2537,      #features of inception-resnet-v2
             hidden_dim=768,
             key_dim=32, #this is the dimension after div by num_heads 
             value_dim=32, 
             num_label=2, 
             num_head=4,
             num_layer=2, 
             mode=opt.mode, #one-shot
             device=device) #cuda
"""



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

        self.que_emb = nn.LSTM(input_size=input_dim, hidden_size=key_dim*self.num_head//2, num_layers=1, bidirectional=True)

        # ans_emb
        if self.mode == 'att-concat-one-shot':
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=key_dim*self.num_head//2, num_layers=1, bidirectional=True)
        else:
            self.ans_emb = nn.LSTM(input_size=input_dim, hidden_size=value_dim*self.num_head//2, num_layers=1, bidirectional=True)



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
        dim = self.num_head * value_dim
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


    
    """predictions, loss = 
                model(question,
                question_lengths,
                answer,
                answer_lengths,
                text,
                text_lengths,
                images,
                image_lengths,
                label)"""
    
    def forward(self,
                 question,
                 question_lengths,
                 answer, answer_lengths,
                       text, 
                       text_lengths, 
                       images, 
                       image_lengths, label):
        
        

        # """
        # Standard LSTMs process every time step in the sequence, including padded positions (i.e., zeros).
        # Using pack_padded_sequence ensures the LSTM only processes non-padded tokens, skipping unnecessary computations.
        # """
         
        # instead of permute use batch_first = True in pack_padded_sequence
        question_embed = pack_padded_sequence(input=question,
                                              lengths=question_lengths,
                                              enforce_sorted=False )
                                            #  ,batch_first=True)
        
        packed_question, (h,c) = self.que_emb(question_embed)
        h = h.permute(1,0,2).contiguous() #(batch, num_layer, hidden_size) #hidden_size=key_dim*self.num_head//2
        question = h.view(h.shape[0], -1) #(batch, num_layer*hidden_size) (bs,2*128)



        answer_embed = pack_padded_sequence(input=answer, 
                                            lengths=answer_lengths,
                                            enforce_sorted=False)
        
        packed_answer, (h,c) = self.ans_emb(answer_embed)
        h = h.permute(1,0,2).contiguous()
        answer = h.view(h.shape[0], -1)

        images   = self.img_emb(images) # (bs*4, max_image_length, hidden_dim) #(4*8,40,768)

        batch_size = question.shape[0] #batch first
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

            sentence_quries = getattr(self,'self_sentence_query'+str(i+1))(text)
            image_quries    = getattr(self,'self_image_query'+str(i+1))(images)   
            queries.append(torch.cat([sentence_quries, image_quries], 1)) # (num_head, bs*4, text_len+image_len, key_dim)
 
            # second copy
            sentence_keys = getattr(self,'self_sentence_key'+str(i+1))(text)
            image_keys    = getattr(self,'self_image_key'+str(i+1))(images)
            keys.append(torch.cat([sentence_keys, image_keys], 1)) # (num_head, bs*4, text_len+image_len, key_dim)

            # third copy
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

        

        # album_feat that will contain the encoded and well presented album contents
        # the context here is the album data ,remember :)
        album_feat = []

        """queries, keys, values = [], [], [] ; are lists each one of 4 items , each item repre just one head."""
        for i in range(self.num_head): #4 times
            """these are the keys , queries and values of the context (images/texts)"""
            """pass the 1st key,query,value to make the self attention""" 

            self_att = self.self_attention(queries[i], keys[i], values[i], mask)
            album_feat.append(self_att) 
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
        
        # query will be the question

        """here is making the attention between the context(images/text)and the question
           ,the KV attention is that key and value comes from the same source ,
            here is the album or the context.
            And the Query comes from another source which is the training data
            which is the Question embdedding  """
        # Projection
                                    #   query     keys        values
        atts_question = self.attention(question, album_keys, album_values, mask) # (bs*4, value_dim)
        
        
        if self.mode == 'one-shot':
            outputs = torch.cat([question,
                                  answer,
                                  atts_question,
                                  answer*atts_question],1) # (bs*4, value_dim*3+key_dim)
            
            outputs = self.answer_proj(outputs) # (bs*4, 2)
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