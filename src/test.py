import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from model import (MemexQA , MemexQA_FVTA)
from dataset import TestDataset, EvalDataset
from test_collate_fn import collate_fn

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

# Use configuration class when you run on notebooks instead argumentparsers
# opt = config()

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

# evaluation mode
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