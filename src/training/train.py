# coding=utf-8

from __future__ import print_function, division
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random

from models.memex import MemexQA
from models.memex_FVTA import MemexQA_FVTA


from dataset import TrainDataset, DevDataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate, default=0.001')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay, default=5e-6')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')
parser.add_argument('--mode', default='one-shot', help='training experiments')
parser.add_argument('--inpf', default='./new_dataset/', help='folder for input data')
parser.add_argument('--outf', default='./output/', help='folder to output csv and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--keep', action='store_true', help='train the model from the previous spec')
parser.add_argument('--FVTA', action='store_true', help='train the model using FVTA')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# Specify cuda
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    device = torch.device("cuda:{}".format(opt.gpu_id))
else:
    device = torch.device("cpu")
    
"""
side note :
if u found item has 7 albums for examples with total images == 64 in the 7 albums
so i will stack the 64 together 
but before stacking the 7 albums together first i padded the photos in each album
so u will have different shape size for # of total photos like 70

photo titles shape
torch.Size([70, 9, 768]) -- is the final shape for 7 albums each one has many images

image feats shape
torch.Size([64, 2537])  #(total num of images , dim)




"""
# collate_fn
def collate_fn(batch):
    # question_embed, answer, choice
    # album_title, album_description, album_when, album_where
    # image_feats, photo_titles, image_lengths
    answer_embed = [item[1] for item in batch] # (bs, answer_len, 768)
    choice_embed = [item[2] for item in batch] # (bs, 3, answer_len, 768)

    answer = [] # 3 ans + 9 other cohices = 12 choices
    for ans, cho in zip(answer_embed, choice_embed):
        answer.append(ans)
        # cho is list of 3 ans
        for c in cho:
            answer.append(c)
    
    # ans.shape[0] is varying in this batch between (3,4 and 5)
    answer_lengths = torch.LongTensor([ans.shape[0] for ans in answer]) # (bs*4, max_len, 768)  
    # print(f'answer_lengths {answer_lengths}')

    # answer = pad_sequence(answer) #torch.Size([5, 12, 768])
    answer = pad_sequence(answer , batch_first= True) #torch.Size([12, 5, 768])
    bs, num_label = len(batch), 4  # number of labels is 4 [ans , c1, c2, c3]


    question_embed = [item[0] for item in batch] # (bs,question_len,768)


    """the most confusing part is here .
       the question is repeated 4 times for (answer , choice1 , choice2 , choice3) """    
    question = []
    for que in question_embed:
        # loop 4 times
        for _ in range(num_label):
            question.append(que)

    question_lengths = torch.LongTensor([que.shape[0] for que in question]) 
    question = pad_sequence(question,batch_first=True)#(bs*4, max_len, 768)
    

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
                tensor = tensor.permute(0,-1,-2)

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

    
    # here as we know the first dimension is variable lenght
    text_lengths = torch.LongTensor([t.shape[0] for t in text]) # (bs*4)


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



    # images 
    images = [item[7] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 2537)
    image_lengths = torch.LongTensor([i.shape[0] for i in images]) # (bs*4)
    images = pad_sequence(images, batch_first=True)


    # Label 
    # put 0 for the correct answer and 1s for the 3 other choices 
    label = torch.LongTensor(bs*([0]+[1]*(num_label-1)))
    """The label tensor is constructed as [0, 1, 1, 1] for each item in the batch,
       repeated for the batch size.
       0 corresponds to the correct answer, and 1 corresponds to the incorrect choices."""


    return( question,  # (bs*4, max_question_len, 768)
            question_lengths,  # (bs*4,)
            answer,  # (bs*4, max_answer_len, 768)
            answer_lengths,  # (bs*4,)
            text,  # (bs*4, max_text_len, 768)
            text_lengths,  # (bs*4,)
            images,  # (bs*4, max_image_len, 2537)
            image_lengths,  # (bs*4,)
            label)  # (bs*4,)


# DataSet and DataLoader
train_dataset = TrainDataset(opt.inpf)
dev_dataset   = DevDataset(opt.inpf)


train = DataLoader(train_dataset, batch_size=opt.batchSize, 
                   shuffle=True, num_workers=opt.workers,
                   collate_fn=collate_fn)

dev   = DataLoader(dev_dataset, batch_size=opt.batchSize,
                   shuffle=False, num_workers=opt.workers,
                   collate_fn=collate_fn)



if opt.FVTA:
    model = MemexQA_FVTA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=32, value_dim=32, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)
else:
    model = MemexQA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=32, value_dim=32, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)



if opt.keep:
    checkpoint = torch.load(f'{opt.outf}/checkpoint_best')
    model.load_state_dict(checkpoint['model_state_dict'])
    last_epoch = checkpoint['epoch']
else:
    last_epoch = 0
model.to(device)
print(model)



# Criterion
# optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adadelta(model.parameters(), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=opt.weight_decay)



if opt.keep:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


Acc = 0.0
train_length, best_loss = len(train), float('inf')

for epoch in range(last_epoch, opt.niter):
    # Training
    train_loss, count = 0.0, 0
    model.train()
    for id, data in enumerate(train):
        question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data
        
        optimizer.zero_grad()
        
        question = question.to(device)
        answer = answer.to(device)
        images  = images.to(device)
        text = text.to(device)
        label = label.to(device)

        predictions, loss = model(question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label)

        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        count += 1

        if (id+1) % 30 == 0:
            print('Epoch :{}, Progress : {}/{}, Loss:{:.3f}, DevLoss:{:.3f}, Acc:{}%'.format(epoch+1, id+1, train_length, train_loss/count, best_loss, int(Acc*100)))
            train_loss = 0.0
            count      = 0


    model.eval()
    dev_loss, dev_count, acc, acc_count = 0., 0, 0., 0
    with torch.no_grad():
        for id, data in enumerate(dev):
            question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data
            
            question = question.to(device)
            answer = answer.to(device)
            images  = images.to(device)
            text = text.to(device)
            label = label.to(device)

            question_lengths = torch.LongTensor(question_lengths)
            answer_lengths = torch.LongTensor(answer_lengths)
            
            predictions, loss = model(question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label)
            predictions = torch.argmax(predictions.view(-1, 4), -1).detach().cpu().numpy()
            
            acc += np.sum(predictions == np.zeros_like(predictions))
            acc_count += label.shape[0]//4

            dev_loss += loss.item()
            dev_count += 1
            
        dev_loss = dev_loss/dev_count
        acc = acc/acc_count

        print(acc)
        if acc < Acc:
            lr_scheduler.step()
        else:
            torch.save({ 'epoch': epoch+1, \
                'model_state_dict': model.state_dict(), \
                'optimizer_state_dict': optimizer.state_dict(), \
                'loss': dev_loss, \
                'acc': acc}, \
                '{}/checkpoint_best'.format(opt.outf))
            Acc = acc

        if dev_loss < best_loss:
            best_loss = dev_loss