import torch
from dataset import TrainDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import random


manualSeed = 1126
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


"""
Word-Based Processing (for question, answer, and choices) → Expands these to match the lengths of text and images.
Sentence-Based Processing (for album info + photo titles) → Maintains original structure for text.
Final Shapes ensure compatibility between question, answer, text, and images.

1.Word-Based Expansion for question, answer, and choices
    --Repeats each question for all choices (num_label = 4).
    --Answers are padded with their choices to match dimensions.
2.Sentence-Based Processing for album info + photo titles
    --Concatenates album_title, album_desp, album_when, 
      album_where, photo_titles along the sequence dimension.

So, at the end u will have a good and compatable dimensions like that :
    question  →  torch.Size([16, 11, 768])    
    answer    →  torch.Size([16, 11, 768])  
    text      →  torch.Size([16, 97, 768])  
    images    →  torch.Size([16, 65, 2537])  
    label     →  torch.Size([16])

    
Another example with bs = 10
    question  →  torch.Size([40, 12, 768])
    answer    →  torch.Size([40, 11, 768])
    text      →  torch.Size([40, 97, 768])
    images    →  torch.Size([40, 65, 2537])
    
    label     →  torch.Size([40])


"""

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
    answer_lengths = torch.LongTensor([ans.shape[0] for ans in answer])
    answer = pad_sequence(answer , batch_first= True)  # (bs*4, max_len, 768)

    bs, num_label = len(batch), 4

    question_embed = [item[0] for item in batch] # (bs, answer_len, 768)
    question = []
    for que in question_embed:
        for _ in range(num_label):
            question.append(que)
    question_lengths = torch.LongTensor([que.shape[0] for que in question]) 
    question = pad_sequence(question , batch_first= True) # (bs*4, max_len, 768)

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

    return(question,  # (bs*4, max_question_len, 768)
            question_lengths,  # (bs*4,)
            answer,  # (bs*4, max_answer_len, 768)
            answer_lengths,  # (bs*4,)
            text,  # (bs*4, max_text_len , 768)
            text_lengths,  # (bs*4,)
            images,  # (bs*4, max_image_len, 2537)
            image_lengths,  # (bs*4,)
            label)  # (bs*4,)



# Dataset Size = 10 
def loader_data():
    path_to_pickle = r'E:\Memex_QA\single_album\prepro'
    train_dataset = TrainDataset(path_to_pickle)
    # print(f'length of train_dataset {len(train_dataset)}')
    train = DataLoader(train_dataset, batch_size =4, 
                       shuffle=True, num_workers=0,
                       collate_fn=collate_fn) 
    return train



def test_model(train):
    from model import MemexQA , MemexQA_FVTA
    device = torch.device("cpu")  
    model = MemexQA(input_dim=768,  #word embedding of bert
                img_dim=2537,      #features of inception-resnet-v2
                hidden_dim=768,
                key_dim=32, #this is the dimension after div by num_heads 
                value_dim=32, 
                num_label=2, 
                num_head=4,
                num_layer=2, 
                mode="one-shot", #one-shot
                device=device) #cuda

    # model = MemexQA_FVTA(input_dim=768,  #word embedding of bert
    #             img_dim=2537,      #features of inception-resnet-v2
    #             hidden_dim=768,
    #             key_dim=32, #this is the dimension after div by num_heads 
    #             value_dim=32, 
    #             num_label=2, 
    #             num_head=4,
    #             num_layer=2, 
    #             mode="one-shot", #one-shot
    #             device=device) #cuda
    

    
    for id, data in enumerate(train):
        print(f'id {id}')
        question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data

        # question = question.to(device)
        # answer = answer.to(device)
        # images  = images.to(device)
        # text = text.to(device)
        # label = label.to(device)

        predictions, loss = model(question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label)
        print(f'predictions {predictions}')
        print(f'loss {loss}')
        break

    """
    predictions tensor([0.5765, 0.5710, 0.5864, 0.5923, 0.5498, 0.5747,
      0.5585, 0.5782, 0.5703, 0.5461, 0.5731, 0.5486, 0.5734, 0.5700, 0.5824, 0.5999],
      grad_fn=<SelectBackward0>)
    loss 0.7657614350318909
    """
        
    ...


if __name__ == '__main__':
    # Load data and print out the first item in the batch
    train = loader_data()

    for id,data in enumerate(train):
        print('verifying.......')
        question, question_lengths, answer, answer_lengths, text, text_lengths, images, image_lengths, label = data
        print(f'question {question.shape}')
        # print(f'question_lengths {question_lengths}')
        print(f'answer {answer.shape}')
        # print(f'answer_lengths {answer_lengths}')
        # print(f'text {[item.shape for item in text]}')
        print(f'text shape {text.shape}')
        # print(f'text_lengths {text_lengths}')
        print(f'images {images.shape}')
        # print(f'image_lengths {image_lengths}')
        # print(f'label {label}')
        print(f'label shape {label.shape}')
        break


    test_model(train=train)






    

    # new output after modifying the dimensionality problem
    """
    # using word_based for QA and Choices 
    starting training.......
    question torch.Size([16, 11, 768])
    question_lengths tensor([ 7,  7,  7,  7, 11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9])
    
    answer torch.Size([16, 11, 768])
    answer_lengths tensor([ 6,  3,  7,  5,  6,  3,  3,  7,  5,  5,  5,  5,  5, 11, 4,  4])
    
    text [torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768])]
    text shape torch.Size([16, 97, 768])
    text_lengths tensor([13, 13, 13, 13, 13, 13, 13, 13, 97, 97, 97, 97, 13, 13, 13, 13])
    
    images torch.Size([16, 65, 2537])
    image_lengths tensor([ 9,  9,  9,  9,  9,  9,  9,  9, 65, 65, 65, 65,  9,  9,  9,  9])
    
    label tensor([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    label shape torch.Size([16])



    # oldd 
    starting training.......
    question torch.Size([16, 768])
    question_lengths tensor([768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768,
            768, 768])
    answer torch.Size([16, 768])
    answer_lengths tensor([768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768,
            768, 768])
    text [torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768]), torch.Size([97, 768])]
    text shape torch.Size([16, 97, 768])
    text_lengths tensor([13, 13, 13, 13, 13, 13, 13, 13, 97, 97, 97, 97, 13, 13, 13, 13])
    images torch.Size([16, 65, 2537])
    image_lengths tensor([ 9,  9,  9,  9,  9,  9,  9,  9, 65, 65, 65, 65,  9,  9,  
    9,  9])
    label tensor([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])

    """






    # old implementation with some dimensionality issues 
    """
    starting training.......
    question torch.Size([11, 16, 768])
    question_lengths tensor([ 7,  7,  7,  7, 11, 11, 11, 11, 10, 10, 10, 10,  9,  9,  9,  9])
    answer torch.Size([11, 16, 768])
    answer_lengths tensor([ 6,  3,  7,  5,  6,  3,  3,  7,  5,  5,  5,  5,  5, 11, 
    4,  4])

    
    max h 72 , max w 13
    # batch level
    # torch.Size([360, 13, 768]) this is one item , repeated it 4 times to be
    # aligned with answers 
    # this is batch of 4 items 
    
    text torch.Size([16, 360, 13, 768])
    text_lengths tensor([ 5,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10,  5,  5,  5,  5, 21, 21,
            21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,  6,  6,  6,  6,         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  1,  1,  1,  1,  1,  1,         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,  9,  9,  9,  9,  9,  9,  9,        72, 72, 72, 72,  9,  9,  9,  9])
    images torch.Size([16, 65, 2537])
    image_lengths tensor([ 9,  9,  9,  9,  9,  9,  9,  9, 65, 65, 65, 65,  9,  9,  
    9,  9])
    label tensor([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    
    
    """


# # collate_fn
# def collate_fn(batch):
#     # question_embed, answer, choice
#     # album_title, album_description, album_when, album_where
#     # image_feats, photo_titles, image_lengths
#     answer_embed = [item[1] for item in batch] # (bs, answer_len, 768)
#     choice_embed = [item[2] for item in batch] # (bs, 3, answer_len, 768)

#     answer = []
#     for ans, cho in zip(answer_embed, choice_embed):
#         answer.append(ans)
#         for c in cho:
#             answer.append(c)
#     answer_lengths = torch.LongTensor([ans.shape[0] for ans in answer]) # (bs*4, max_len, 768)
#     answer = pad_sequence(answer)

#     bs, num_label = len(batch), 4

#     question_embed = [item[0] for item in batch] # (bs, answer_len, 768)
#     question = []
#     for que in question_embed:
#         for _ in range(num_label):
#             question.append(que)
#     question_lengths = torch.LongTensor([que.shape[0] for que in question]) 
#     question = pad_sequence(question) # (bs*4, max_len, 768)

#     album_title  = [item[3] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768) .unsqueeze(0).repeat(num_label,1,1)
#     album_desp   = [item[4] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
#     album_when   = [item[5] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
#     album_where  = [item[6] for item in batch for _ in range(num_label)] # (bs*4, num_album, 768)
#     photo_titles = [item[8] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 768)

#     def pad_tensor(tensor, max_dims):
#         """Pads a tensor to match max_dims along the first two dimensions."""
#         target_h, target_w = max_dims  # First two dimensions
#         h, w, d = tensor.shape  # Last dim is 768, which stays the same
#         pad_h = target_h - h
#         pad_w = target_w - w
#         # Pad only along the first two dimensions
#         return F.pad(tensor, (0, 0, 0, pad_w, 0, pad_h))


#     # Get max height & width across all tensors
#     all_tensors = album_title + album_desp + album_when + album_where + photo_titles
    
#     max_h = max(t.shape[0] for t in all_tensors)
#     max_w = max(t.shape[1] for t in all_tensors)

#     # text_lengths = torch.LongTensor([t.shape[0] for t in text]) # (bs*4)
#     text_lengths = torch.LongTensor([t.shape[0] for t in all_tensors]) # (bs*4)
#     print(f'max h {max_h} , max w {max_w}')

#     padded_tensors = [[pad_tensor(t, (max_h, max_w)) for t in lst] for lst in [album_title, album_desp, album_when, album_where, photo_titles]]
#     text = [torch.cat([a, b, c, d, e], dim=0) for a, b, c, d, e in zip(*padded_tensors)]
#     # text = [torch.cat([a,b,c,d,e], 0) for a,b,c,d,e in zip(album_title,album_desp,album_when,album_where,photo_titles)] # (bs, ..., 768)
#     text = pad_sequence(text, batch_first=True) #to be returned as one block


#     images = [item[7] for item in batch for _ in range(num_label)] # (bs*4, num_album*num_photo, 2537)
#     image_lengths = torch.LongTensor([i.shape[0] for i in images]) # (bs*4)
#     images = pad_sequence(images, batch_first=True)

#     # Label smoothing
#     label = torch.LongTensor(bs*([0]+[1]*(num_label-1)))



#     return(question,  # (bs*4, max_question_len, 768)
#         question_lengths,  # (bs*4,)
#         answer,  # (bs*4, max_answer_len, 768)
#         answer_lengths,  # (bs*4,)
#         text,  # (bs*4, max_sentences_len, max_words_len, 768)
#         text_lengths,  # (bs*4,)
#         images,  # (bs*4, max_image_len, 2537)
#         image_lengths,  # (bs*4,)
#         label)  # (bs*4,)

