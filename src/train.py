import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random

from model import MemexQA, MemexQA_FVTA
from test_collate_fn import collate_fn
from configuration import config 

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

# Use configuration class when you run on notebooks instead argumentparsers
# opt = config()

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
            print(f'Epoch :{epoch+1}, Progress : {id+1}/{train_length}, Loss:{train_loss/count:.3f}, DevLoss:{best_loss:.3f}, Acc:{int(Acc*100)}%')
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
                f'{opt.outf}/checkpoint_best')
            
            Acc = acc

        if dev_loss < best_loss:
            best_loss = dev_loss