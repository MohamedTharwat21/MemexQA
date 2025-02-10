# coding=utf-8

from __future__ import print_function, division
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random

from tqdm import tqdm

from models.memex import MemexQA
from models.memex_FVTA import MemexQA_FVTA


from dataset import TestDataset, EvalDataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--manualSeed', type=int, default=1126, help='manual seed')
parser.add_argument('--mode', default='one-shot', help='test experiments')
parser.add_argument('--inpf', default='./new_dataset/', help='folder for input data')
parser.add_argument('--outf', default='./output/', help='folder to output csv and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--FVTA', action='store_true', help='train the model using FVTA')

opt = parser.parse_args()
print(opt)


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
    

# collate_fn
def collate_fn(batch):
    # question_embed, answer, choice
    # album_title, album_description, album_when, album_where
    # image_feats, photo_titles, image_lengths
    answer_embed = [item[1] for item in batch] # (bs, answer_len, 768)
    choice_embed = [item[2] for item in batch] # (bs, 3, answer_len, 768)

    answer = []
    for ans, cho in zip(answer_embed, choice_embed):
        answer.append(ans)
        for c in cho:
            answer.append(c)
    answer_lengths = torch.LongTensor([ans.shape[0] for ans in answer]) # (bs*4, max_len, 768)
    answer = pad_sequence(answer)

    bs, num_label = len(batch), 4

    question_embed = [item[0] for item in batch] # (bs, answer_len, 768)
    question = []
    for que in question_embed:
        for _ in range(num_label):
            question.append(que)
    question_lengths = torch.LongTensor([que.shape[0] for que in question]) 
    question = pad_sequence(question) # (bs*4, max_len, 768)

    album_title  = [item[3] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768) .unsqueeze(0).repeat(num_label,1,1)
    album_desp   = [item[4] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    album_when   = [item[5] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    album_where  = [item[6] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
    photo_titles = [item[8] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 768)
    text = [torch.cat([a,b,c,d,e], 0) for a,b,c,d,e in zip(album_title,album_desp,album_when,album_where,photo_titles)] # (bs, ..., 768)
    text_lengths = torch.LongTensor([t.shape[0] for t in text]) # (bs*4)
    text = pad_sequence(text, batch_first=True)

    images = [item[7] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 2537)
    image_lengths = torch.LongTensor([i.shape[0] for i in images]) # (bs*4)
    images = pad_sequence(images, batch_first=True)

    # Label smoothing
    label = torch.LongTensor(bs*([0]+[1]*(num_label-1)))

    return question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label


# DataSet and DataLoader
# test_dataset = TestDataset(opt.inpf)
test_dataset = TestDataset(opt.inpf)

test = DataLoader(test_dataset, batch_size=opt.batchSize, 
                   shuffle=False, num_workers=opt.workers,
                   collate_fn=collate_fn)

if opt.FVTA:
    model = MemexQA_FVTA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=32, value_dim=32, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)
else:
    model = MemexQA(input_dim=768, img_dim=2537, hidden_dim=768, key_dim=32, value_dim=32, num_label=2, num_head=4, num_layer=2, mode=opt.mode, device=device)
checkpoint = torch.load('{}/checkpoint_best'.format(opt.outf))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print(model)

model.eval()
acc, acc_count = 0., 0
with torch.no_grad():
    for id, data in tqdm(enumerate(test)):
        question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data
            
        question = question.to(device)
        answer = answer.to(device)
        images  = images.to(device)
        text = text.to(device)
        label = label.to(device)
            
        predictions, loss = model(question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label)
        predictions = torch.argmax(predictions.view(-1, 4), -1).detach().cpu().numpy()

        acc += np.sum(predictions == np.zeros_like(predictions))
        acc_count += label.shape[0]//4

    acc = acc/acc_count
    print('Testing Accuracy : {:3f}%'.format(100*acc))