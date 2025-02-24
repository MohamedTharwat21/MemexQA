import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random


# from model import MemexQA, MemexQA_FVTA
# from dataset import TrainDataset, DevDataset
# from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from use_dataset import train_dataset


# # collate_fn
def collate_fn(batch):
    # question_embed, answer, choice
    # album_title, album_description, album_when, album_where
    # image_feats, photo_titles, image_lengths
    answer_embed = [item[1] for item in batch] # (bs, answer_len, 768)
    choice_embed = [item[2] for item in batch] # (bs, 3, answer_len, 768)
     
    
    # print([it.shape for it in answer_embed])
    # [torch.Size([4, 768]), torch.Size([3, 768]), torch.Size([4, 768])]
    
    # print([itt.shape for item in choice_embed for itt in item])
    # 3 for each element in the batch
    # [torch.Size([5, 768]), torch.Size([4, 768]), torch.Size([4, 768]), 
    #  torch.Size([3, 768]), torch.Size([3, 768]), torch.Size([3, 768]), 
    #  torch.Size([4, 768]), torch.Size([4, 768]), torch.Size([3, 768])]

    answer = [] # 3 ans + 9 other cohices = 12 choices
    for ans, cho in zip(answer_embed, choice_embed):
        answer.append(ans)
        # cho is list of 3 ans
        for c in cho:
            answer.append(c)

    
    # answer = [item.unsqueeze(0) for item in answer]

    # print("surprise...")
    # print(f'length of answer list {len(answer)}') #12
    # for _ans in answer :
    #     print(_ans.shape)



    # ans.shape[0] is varying in this batch between (3,4 and 5)
    answer_lengths = torch.LongTensor([ans.shape[0] for ans in answer]) # (bs*4, max_len, 768)  
    # print(f'answer_lengths {answer_lengths}')

    # answer = pad_sequence(answer) #torch.Size([5, 12, 768])
    answer = pad_sequence(answer , batch_first= True) #torch.Size([12, 5, 768])

    # print(f'answer list shape after padding {answer.shape}')  
    # for _ans in answer :
    #     print(_ans.shape)
  


    bs, num_label = len(batch), 4  # number of labels is 4 [ans , c1, c2, c3]


    question_embed = [item[0] for item in batch] # (bs,question_len,768)
    # question_embed = [item.unsqueeze(0) for item in question_embed]
    # print('surprise...')
    # for qe in question_embed:
    #     print(qe.shape)
        # """ torch.Size([8, 768])
        #     torch.Size([10, 768])
        #     torch.Size([11, 768])"""

    
    

    """the most confusing part is here .
       the question is repeated 4 times for (answer , choice1 , choice2 , choice3) """
    
    question = []
    for que in question_embed:
        # loop 4 times
        for _ in range(num_label):
            question.append(que)
    
    # print(f'len(question) : {len(question)}') #12


    question_lengths = torch.LongTensor([que.shape[0] for que in question]) 
    question = pad_sequence(question,batch_first=True)#(bs*4, max_len, 768)
    
    # print(f'question shape: {question.shape}') # torch.Size([12, 11, 768])


     

    # u have batch of 3 items then u have 3 album_titles 
    # for every album title repeat it 4 times 
    # so album_title list will be 12 item

    album_title  = [item[3] for item in batch for _ in range(num_label)] # (bs*4, title_len, 768) .unsqueeze(0).repeat(num_label,1,1)
    album_desp   = [item[4] for item in batch for _ in range(num_label)] # (bs*4, descrp_len , 768)
    album_when   = [item[5] for item in batch for _ in range(num_label)] # (bs*4, when_len, 768)
    album_where  = [item[6] for item in batch for _ in range(num_label)] # (bs*4, where_len, 768)
    photo_titles = [item[8] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 768)
    
   
    # context 
    # text = [torch.cat([a,b,c,d,e], 0) for a,b,c,d,e in zip(album_title,album_desp,album_when,album_where,photo_titles)] # (bs, ..., 768)
    #     text = [torch.cat([a,b,c,d,e], 0) for a,b,c,d,e in zip(album_title,album_desp,album_when,album_where,photo_titles)] # (bs, ..., 768)
    
    text = []

    for a, b, c, d, e in zip(album_title, album_desp, album_when, album_where, photo_titles):
        # Ensure all tensors have 3 dimensions
        def ensure_three_dims(tensor):
            if tensor.dim() == 2:  # If the tensor has only 2 dimensions, add a third one
                tensor = tensor.unsqueeze(0)  # Add a new dimension at the front
            return tensor

        a, b, c, d, e = map(ensure_three_dims, [a, b, c, d, e])

        # Find the maximum lengths along dimensions 0 and 1
        max_len_dim0 = max(a.size(0), b.size(0), c.size(0), d.size(0), e.size(0))
        max_len_dim1 = max(a.size(1), b.size(1), c.size(1), d.size(1), e.size(1))
        fixed_dim2 = a.size(2)  # Assuming the last dimension is fixed for all tensors

        # Pad each tensor to the maximum sizes
        def pad_tensor(tensor, max_dim0, max_dim1, fixed_dim2):
            if tensor.size(-2) == 768 :
                # print(tensor.shape)
                tensor = tensor.permute(0,-1,-2)
                # print('permuting .... ')
 
            pad_dim0 = max_dim0 - tensor.size(0)
            pad_dim1 = max_dim1 - tensor.size(1)
            padded_tensor = tensor
            if pad_dim0 > 0:
                padded_tensor = torch.cat([padded_tensor, torch.zeros(pad_dim0, tensor.size(1), fixed_dim2)], dim=0)
            if pad_dim1 > 0:
                padded_tensor = torch.cat([padded_tensor, torch.zeros(padded_tensor.size(0), pad_dim1, fixed_dim2)], dim=1)
            return padded_tensor

        a_padded = pad_tensor(a, max_len_dim0, max_len_dim1, fixed_dim2)
        b_padded = pad_tensor(b, max_len_dim0, max_len_dim1, fixed_dim2)
        c_padded = pad_tensor(c, max_len_dim0, max_len_dim1, fixed_dim2)
        d_padded = pad_tensor(d, max_len_dim0, max_len_dim1, fixed_dim2)
        e_padded = pad_tensor(e, max_len_dim0, max_len_dim1, fixed_dim2)

        # Concatenate the padded tensors along dimension 0
        concatenated = torch.cat([a_padded, b_padded, c_padded, d_padded, e_padded], dim=0)
        
        text.append(concatenated)

    
    # for item in text :
    #     print(f'final item in text shape {item.shape}')


    
    # here as we know the first dimension is variable lenght
    text_lengths = torch.LongTensor([t.shape[0] for t in text]) # (bs*4)
    # print("*" * 100)
    # print(f'text lengths {text_lengths}')

 
    
    def pad_to_max_size(tensors, max_dim0, max_dim1, fixed_dim2):
        """
        Pads a list of tensors to the specified maximum sizes.
        """
        padded_tensors = []
        for tensor in tensors:
            # Compute padding sizes
            pad_dim0 = max_dim0 - tensor.size(0)
            pad_dim1 = max_dim1 - tensor.size(1)
            # Pad tensor
            padded_tensor = tensor
            if pad_dim0 > 0:
                padded_tensor = torch.cat([padded_tensor, torch.zeros(pad_dim0, tensor.size(1), fixed_dim2)], dim=0)
            if pad_dim1 > 0:
                padded_tensor = torch.cat([padded_tensor, torch.zeros(padded_tensor.size(0), pad_dim1, fixed_dim2)], dim=1)
            padded_tensors.append(padded_tensor)
        return torch.stack(padded_tensors, dim=0)
    

    # Determine maximum dimensions across all tensors
    max_dim0 = max(tensor.size(0) for tensor in text)
    max_dim1 = max(tensor.size(1) for tensor in text)
    fixed_dim2 = text[0].size(2)  # Assuming the last dimension is fixed for all tensors

    # Pad all tensors in the list to the same dimensions
    text_padded = pad_to_max_size(text, max_dim0, max_dim1, fixed_dim2) #torch.Size([12, 45, 107, 768])

    # print(f"Padded text shape: {text_padded.shape}") #torch.Size([12, 45, 107, 768])
        



    images = [item[7] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 2537)
    image_lengths = torch.LongTensor([i.shape[0] for i in images]) # (bs*4)
    images = pad_sequence(images, batch_first=True)

    # Label smoothing
        # put 0 for the correct answer and 1s for the 3 other choices 
    """
    for example batch size = 8  
    label tensor([  0, 1, 1, 1,
                    0, 1, 1, 1,
                    0, 1, 1, 1,
                    0, 1, 1, 1,
                    0, 1, 1, 1,
                    0, 1, 1, 1,  
                    0, 1, 1, 1,
                    0, 1, 1, 1]) 
    """

    label = torch.LongTensor(bs*([0]+[1]*(num_label-1)))



    return( question,  # (bs*4, max_question_len, 768)
            question_lengths,  # (bs*4,)
            answer,  # (bs*4, max_answer_len, 768)
            answer_lengths,  # (bs*4,)
            text,  # (bs*4, max_text_len, 768)
            text_lengths,  # (bs*4,)
            images,  # (bs*4, max_image_len, 2537)
            image_lengths,  # (bs*4,)
            label)  # (bs*4,)





def loader_data():
    from use_dataset import train_dataset
    from torch.utils.data import DataLoader

    train = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0,
                    collate_fn=collate_fn) 
    return train



"""
    question_embed.detach(),\
    answer_embed.detach(),  \
    [cho.detach() for cho in choices_embed], \
    
    album_title.detach(), \
    album_description.detach(), \
    album_when.detach(), \
    album_where.detach(), \
    
    image_feats.detach(), \
    photo_titles.detach(), \
    image_lengths.detach()
"""


if __name__ == '__main__':
      


    # Load data and print out the first item in the batch
    train = loader_data()
    for id,data in enumerate(train):
        print('starting training.......')
        question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data
        print(f'question {question.shape}')
        print(f'question_lengths {question_lengths}')
        print(f'answer {answer.shape}')
        print(f'answer_lengths {answer_lengths}')
        print(f'text {[item.shape for item in text]}')
        print(f'text_lengths {text_lengths}')
        print(f'images {images.shape}')
        print(f'image_lengths {image_lengths}')
        print(f'label {label}')
        break



    """
    # the item itself it's like batch of 8 items 
    # as we create the same data for each choice 
                        # 32 = 4 * 8 (bs)
    question torch.Size([32, 14, 768])
    ~question_lengths tensor([  12, 12, 12, 12,
                                9,  9,  9,  9,
                                11, 11, 11, 11,
                                14, 14, 14, 14,
                                12, 12, 12, 12, 
                                9,  9,  9,  9, 
                                6,  6,  6,  6,
                                10, 10, 10, 10])

    answer torch.Size([32, 10, 768])
    ~answer_lengths tensor([ 4,  3,  4,  3,
                             3,  5,  4,  3,
                             5,  3, 10,  5,
                             4,  3,  4,  7, 
                             3,  3, 3,  3, 
                             7,  3,  3,  3,
                             3,  3,  4,  5, 
                             4,  4,  7,  5])

    
    # torch.Size([200, 12, 768]) --> this is just one item in the batch
    # this is representing the context {text} for this item
    # i repeated it 4 times , one for each choice 


    text [torch.Size([200, 12, 768]), torch.Size([200, 12, 768]),
          torch.Size([200, 12, 768]), torch.Size([200, 12, 768]), 

          torch.Size([40, 10, 768]), torch.Size([40, 10, 768]),
          torch.Size([40, 10, 768]), torch.Size([40, 10, 768]),

          torch.Size([250, 37, 768]), torch.Size([250, 37, 768]), 
          torch.Size([250, 37, 768]), torch.Size([250, 37, 768]), 

          torch.Size([200, 78, 768]), torch.Size([200, 78, 768]), 
          torch.Size([200, 78, 768]), torch.Size([200, 78, 768]), 

          torch.Size([40, 7, 768]), torch.Size([40, 7, 768]),
          torch.Size([40, 7, 768]), torch.Size([40, 7, 768]),

          torch.Size([30, 11, 768]), torch.Size([30, 11, 768]),
          torch.Size([30, 11, 768]), torch.Size([30, 11, 768]),

          torch.Size([40, 71, 768]), torch.Size([40, 71, 768]), 
          torch.Size([40, 71, 768]), torch.Size([40, 71, 768]), 

          torch.Size([300, 147, 768]), torch.Size([300, 147, 768]),
          torch.Size([300, 147, 768]), torch.Size([300, 147, 768])]

 
    ~text_lengths tensor([200, 200, 200, 200,
                           40,  40,  40,  40, 
                           250, 250, 250, 250,
                           200, 200,200, 200, 
                           40,  40,  40,  40, 
                           30,  30,  30,  30,
                           40,  40,  40,  40,
                           300, 300, 300, 300])

    images torch.Size([32, 50, 2537])
    ~image_lengths tensor([37, 37, 37, 37,
                            8,  8,  8,  8,
                            42, 42, 42, 42,
                            33, 33, 33, 33,  
                            8,  8, 8,  8,
                            6,  6,  6,  6, 
                            8,  8,  8,  8,
                            50, 50, 50, 50])
    
    label tensor([0, 1, 1, 1,
                  0, 1, 1, 1,
                  0, 1, 1, 1,
                  0, 1, 1, 1,
                  0, 1, 1, 1,
                  0, 1, 1, 1,  
                  0, 1, 1, 1,
                  0, 1, 1, 1])
    """




""""iconic error : 
    RuntimeError: [enforce fail at alloc_cpu.cpp:114] data. 
    DefaultCPUAllocator: not enough memory: you tried to allocate 9909043200 bytes."""


# batch_indices = list(range(10)) # Adjust indices based on your dataset size
# test_batch = [train_dataset[idx] for idx in batch_indices]
 
# # print(test_batch[0])
# output = collate_fn(test_batch)
# question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = output
 

# print("test collate function............")
# print(f"Question shape: {question.shape}")
# print(" ")
# # print(f"Question lengths: {question_lengths}")
# print(f"Answer shape: {answer.shape}")
# print(" ")
# # print(f"Answer lengths: {answer_lengths}")
# print(f"Text shape: {[item.shape for item in text]}")
# print(" ")
# # print(f"Text lengths: {text_lengths}")
# print(f"Images shape: {images.shape}")
# print(" ")
# # print(f"Image lengths: {image_lengths}")
# print(f"Label shape: {label.shape}")
