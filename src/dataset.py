import os
import pickle
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TrainDataset(Dataset):
    def __init__(self, prepropath):
        super(TrainDataset, self).__init__()
        data_path = os.path.join(prepropath,"Unified_multi_albums_data.p")
        shared_path = os.path.join(prepropath,"Unified_multi_albums_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']
        
    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], self.data['choices'][idx]
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])     
            album_description.append(album['description'])
            album_where.append(album['where'])           
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
     
            # padded_photo_titles = pad_sequence(album['photo_titles'], batch_first=True)
            # photo_titles.append(padded_photo_titles)
 
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        

        
        album_title = torch.stack(album_title)
        # album_title = pad_sequence(album_title, batch_first=True)

        album_description = torch.stack(album_description)
        # album_description = pad_sequence(album_description, batch_first=True)

        album_when = torch.stack(album_when)
        # album_when = pad_sequence(album_when, batch_first=True)

        album_where = torch.stack(album_where)
        # album_where = pad_sequence(album_where, batch_first=True)
        
        
        # photo titles 
        # print([p.shape for p in photo_titles])
        # photo_titles = torch.cat(photo_titles, 0)
        #         print(f"index/id {idx}")
        # print(f"photo titles shapes {[item.shape for item in photo_titles]}")

        # photo_titles = torch.cat(photo_titles, 0) 
        # cat on dim = 0 , but there is a difference in the second dim also

        # Step 1: Find max sequence count (first dimension)
        # max_seq_count = max(title.shape[0] for title in photo_titles)  # Max number of sequences (e.g., 9)

        # # Step 2: Find max token count (second dimension)
        # max_token_count = max(title.shape[1] for title in photo_titles)  # Max sequence length (e.g., 13)

        # # Step 3: Pad each tensor to match (max_seq_count, max_token_count, 768)
        # padded_titles = []
        # for title in photo_titles:
        #     padded_title = torch.nn.functional.pad(
        #         title, (0, 0, 0, max_token_count - title.shape[1], 0, max_seq_count - title.shape[0])
        #     )  # Pad along dim=1 (tokens) and dim=0 (sequences)
        #     padded_titles.append(padded_title)
        
        # # Neverrrrrr Stack
        # # Step 4: Stack into a single tensor
        # # photo_titles = torch.stack(padded_titles)  # Shape: (batch_size, max_seq_count, max_token_count, 768)
        # # print(photo_titles.shape)  # Should be (8, 9, 13, 768)

        # # there is a great difference
        # # Step 4: Cat into a single tensor
        # photo_titles = torch.cat(padded_titles)  # Shape: (batch_size*max_seq_count , max_token_count, 768)
        # # print(f'photo titles shape { photo_titles.shape }')  # Should be (8*9, 13, 768)
      

        photo_titles = torch.cat(photo_titles, 0)

        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(),\
               answer.detach(), \
               [cho.detach() for cho in choice], \
               album_title.detach(),\
               album_description.detach(),\
               album_when.detach(),\
               album_where.detach(), \
               image_feats.detach(),\
               photo_titles.detach(),\
               image_lengths.detach()

    def __len__(self):
        return self.num_examples


# class TrainDataset(Dataset):
#     def __init__(self, prepropath):
#         super(TrainDataset, self).__init__()

#         data_path = os.path.join(prepropath,"Unified_multi_albums_data.p")
#         shared_path = os.path.join(prepropath,"Unified_multi_albums_shared.p")

#         with open(data_path, "rb")as fp:
#             self.data = pickle.load(fp)
#         with open(shared_path, "rb") as fp:
#             self.shared = pickle.load(fp)

#         self.num_examples = len(self.data['questions'])

#         self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']
        

#     # replace torch.stack() with pad_sequence()
#     def __getitem__(self,idx):
#         question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
#         albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], self.data['choices'][idx]
#         yidx = self.data['yidx'][idx]

#         # Get image features
#         # A album has a lot of infomation and series of images
#         ## shared
#         album_title, album_description = [], []
#         album_where, album_when = [], []
#         ## Respectively
#         photo_titles = []
#         image_feats, image_lengths = [], []
#         for albumId in albumIds:
#             album = self.shared['albums'][albumId]
#             album_title.append(album['title'])
#             album_description.append(album['description'])
#             album_where.append(album['where'])
#             album_when.append(album['when'])

#             padded_photo_titles = pad_sequence(album['photo_titles'], batch_first=True)
#             photo_titles.append(padded_photo_titles)
#             # photo_titles.append(torch.stack(album['photo_titles']))
            
#             """focus : len of this list is 8 , each one is album
#                shape (sentences,max_words,768) 
            
#             """
#             # idx 0 (first album):
#             # photo titles shapes [torch.Size([9, 13, 768]), torch.Size([8, 12, 768]),
#             #                      torch.Size([8, 7, 768]), torch.Size([9, 7, 768]), 
#             #                      torch.Size([9, 4, 768]), torch.Size([8, 6, 768]), 
#             #                      torch.Size([8, 7, 768]), torch.Size([6, 11, 768])]
           
#             image_feat = []
#             for pid in album['photo_ids']:
#                 image_feat.append(self.shared['pid2feat'][pid])
            
#             image_feats.append(np.stack(image_feat))
#             image_lengths.append(len(image_feat))
        
#         # album_title = torch.stack(album_title)
#         album_title = pad_sequence(album_title)
#         # album_description = torch.stack(album_description)
#         album_description = pad_sequence(album_description)

#         # album_when = torch.stack(album_when)
#         album_when = pad_sequence(album_when)

#         # album_where = torch.stack(album_where)
#         album_where = pad_sequence(album_where)
        

          #photo titles 
#         print(f"index/id {idx}")
#         # print(f"photo titles shapes {[item.shape for item in photo_titles]}")

#         # photo_titles = torch.cat(photo_titles, 0) 
#         # cat on dim = 0 , but there is a difference in the second dim also

#         # Step 1: Find max sequence count (first dimension)
#         max_seq_count = max(title.shape[0] for title in photo_titles)  # Max number of sequences (e.g., 9)

#         # Step 2: Find max token count (second dimension)
#         max_token_count = max(title.shape[1] for title in photo_titles)  # Max sequence length (e.g., 13)

#         # Step 3: Pad each tensor to match (max_seq_count, max_token_count, 768)
#         padded_titles = []
#         for title in photo_titles:
#             padded_title = torch.nn.functional.pad(
#                 title, (0, 0, 0, max_token_count - title.shape[1], 0, max_seq_count - title.shape[0])
#             )  # Pad along dim=1 (tokens) and dim=0 (sequences)
#             padded_titles.append(padded_title)
        
#         # Neverrrrrr Stack
#         # Step 4: Stack into a single tensor
#         # photo_titles = torch.stack(padded_titles)  # Shape: (batch_size, max_seq_count, max_token_count, 768)
#         # print(photo_titles.shape)  # Should be (8, 9, 13, 768)

#         # there is a great difference
#         # Step 4: Cat into a single tensor
#         photo_titles = torch.cat(padded_titles)  # Shape: (batch_size*max_seq_count , max_token_count, 768)
#         # print(photo_titles.shape)  # Should be (8*9, 13, 768)


#         image_lengths = torch.LongTensor(image_lengths)
#         image_feats = torch.Tensor(np.concatenate(image_feats, 0))

#         return question_embed.detach(),\
#                answer.detach(), \
#                [cho.detach() for cho in choice], \
#                album_title.detach(),\
#                album_description.detach(),\
#                album_when.detach(),\
#                album_where.detach(), \
#                image_feats.detach(),\
#                photo_titles.detach(),\
#                image_lengths.detach()

#     def __len__(self):
#         return self.num_examples


class DevDataset(Dataset):
    def __init__(self, prepropath):
        super(DevDataset, self).__init__()
        data_path = os.path.join(prepropath,"val_data.p")
        shared_path = os.path.join(prepropath,"val_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']

    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], self.data['choices'][idx]
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), [cho.detach() for cho in choice], \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()

    def __len__(self):
        return self.num_examples


class TestDataset(Dataset):
    def __init__(self, prepropath):
        super(TestDataset, self).__init__()
        data_path = os.path.join(prepropath,"test_data.p")
        shared_path = os.path.join(prepropath,"test_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.num_examples = len(self.data['questions'])

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']

    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], self.data['choices'][idx]
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), [cho.detach() for cho in choice], \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()
        
    def __len__(self):
        return self.num_examples


class EvalDataset(Dataset):
    def __init__(self, prepropath):
        super(EvalDataset, self).__init__()
        data_path = os.path.join(prepropath,"test_data.p")
        shared_path = os.path.join(prepropath,"test_shared.p")

        with open(data_path, "rb")as fp:
            self.data = pickle.load(fp)
        with open(shared_path, "rb") as fp:
            self.shared = pickle.load(fp)

        self.keys = ['questions', 'questions_embed', 'answers', 'yidx', 'aid', 'qid', 'choices']

        output = {'questions':[], 'questions_embed':[], 'answers':[], 'yidx':[], 'aid':[], 'qid':[], 'choices':[]}
        for idx, question in enumerate(self.data['questions']):
            if question in ['What was the girl wearing on her eyes?', \
                            'Where was the gravesite?', \
                            'How many People posed for a photo?', \
                            'What color ribbons were around the diplomas?', \
                            'How many teddy bears did we win?', \
                            'Once again it was time for the Turkey Day Run in Downtown Charleston. Itwas the first race of the season with a downtown course, so I had a chance to run part of my regular downtown course without having to worry about traffic.']:
                for key in self.keys:
                    output[key].append(self.data[key][idx])
        self.data = output
                            
        self.num_examples = len(self.data['questions'])

    def __getitem__(self,idx):
        question, question_embed, answer = self.data['questions'][idx], self.data['questions_embed'][idx], self.data['answers'][idx]
        albumIds, qid, choice = self.data['aid'][idx], self.data['qid'][idx], self.data['choices'][idx]
        yidx = self.data['yidx'][idx]

        # Get image features
        # A album has a lot of infomation and series of images
        ## shared
        album_title, album_description = [], []
        album_where, album_when = [], []
        ## Respectively
        photo_titles = []
        image_feats, image_lengths = [], []
        for albumId in albumIds:
            album = self.shared['albums'][albumId]
            album_title.append(album['title'])
            album_description.append(album['description'])
            album_where.append(album['where'])
            album_when.append(album['when'])
            photo_titles.append(torch.stack(album['photo_titles']))
            image_feat = []
            for pid in album['photo_ids']:
                image_feat.append(self.shared['pid2feat'][pid])
            image_feats.append(np.stack(image_feat))
            image_lengths.append(len(image_feat))
        
        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)

        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)

        photo_titles = torch.cat(photo_titles, 0)
        image_lengths = torch.LongTensor(image_lengths)
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))

        return question_embed.detach(), answer.detach(), [cho.detach() for cho in choice], \
               album_title.detach(), album_description.detach(), album_when.detach(), album_where.detach(), \
               image_feats.detach(), photo_titles.detach(), image_lengths.detach()
        
    def __len__(self):
        return self.num_examples

   
if __name__ == '__main__':
    path_to_pickle = r'E:\Memex_QA\single_album\prepro'

    train_dataset = TrainDataset(path_to_pickle)
    print(f'length of train_dataset {len(train_dataset)}')
   
    question_embed, answer, choice, \
    album_title, album_description, album_when, \
    album_where, image_feats, \
    photo_titles, image_lengths = train_dataset[2]
   

    print(f'question_embed {question_embed.shape}')
    print(f'choices shape {[ch.shape for ch in choice]}')
    print(f'album_title {album_title.shape}')
    print(f'album_description {album_description.shape}')
    print(f'album_when {album_when.shape}')
    print(f'album_where {album_where.shape}')
    print(f'image_feats {image_feats.shape}')
    # print(f'photo_titles {[pt.shape for pt in photo_titles]}')  # If photo_titles is a list of tensors
    print(f'photo titles shape {photo_titles.shape}')   
    print(f'image_lengths {image_lengths.shape}')
    exit()


    # dev_dataset   = DevDataset('./new_dataset/')    
    # test_dataset  = TestDataset('./new_dataset/')
    
    # print(train_dataset[0])  
    # print(dev_dataset[0])    
    # print(test_dataset[0])



    """output"""


    # final using new embedding
    # multiple_albums
    """
    # using word_based for QA and Choices 
    length of train_dataset 10
    question_embed torch.Size([10, 768])
    choices shape [torch.Size([5, 768]), torch.Size([5, 768]), torch.Size([5, 768])]
    album_title torch.Size([8, 768])
    album_description torch.Size([8, 768])
    album_when torch.Size([8, 768])
    album_where torch.Size([8, 768])
    image_feats torch.Size([65, 2537])
    photo titles shape torch.Size([65, 768])
    image_lengths torch.Size([8])



    length of train_dataset 10
    question_embed torch.Size([768])
    choices shape [torch.Size([768]), torch.Size([768]), torch.Size([768])]
    album_title torch.Size([8, 768])
    album_description torch.Size([8, 768])
    album_when torch.Size([8, 768])
    album_where torch.Size([8, 768])
    image_feats torch.Size([65, 2537])
    photo titles shape torch.Size([65, 768])
    image_lengths torch.Size([8])
    """

    # single_albums
    """
    # using word_based for QA and Choices 
    length of train_dataset 10
    question_embed torch.Size([9, 768])
    choices shape [torch.Size([11, 768]), torch.Size([4, 768]), torch.Size([4, 768])]
    album_title torch.Size([1, 768])
    album_description torch.Size([1, 768])
    album_when torch.Size([1, 768])
    album_where torch.Size([1, 768])
    image_feats torch.Size([9, 2537])
    photo titles shape torch.Size([9, 768])
    image_lengths torch.Size([1])


    length of train_dataset 10
    question_embed torch.Size([768])
    choices shape [torch.Size([768]), torch.Size([768]), torch.Size([768])]        
    album_title torch.Size([1, 768])
    album_description torch.Size([1, 768])
    album_when torch.Size([1, 768])
    album_where torch.Size([1, 768])
    image_feats torch.Size([9, 2537])
    photo titles shape torch.Size([9, 768])
    image_lengths torch.Size([1])
    
    """

    







    # old implementation
    # single album 
    """    
    length of train_dataset 10
    index 9
    photo titles shapes [torch.Size([9, 13, 768])]
    question_embed torch.Size([9, 768])
    choices shape [torch.Size([11, 768]), torch.Size([4, 768]), torch.Size([4, 768])]
    album_title torch.Size([5, 1, 768])
    album_description torch.Size([21, 1, 768])
    album_when torch.Size([6, 1, 768])
    album_where torch.Size([768, 1])
    image_feats torch.Size([9, 2537])
    photo_titles [torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768])]
    image_lengths torch.Size([1])
    """

    # Multiple albums
    """
    length of train_dataset 10
    index 2
    photo titles shapes [torch.Size([9, 13, 768]), torch.Size([8, 12, 768]), torch.Size([8, 7, 768]), torch.Size([9, 7, 768]), torch.Size([9, 4, 768]), torch.Size([8, 6, 768]), torch.Size([8, 7, 768]), torch.Size([6, 11, 768])]
    question_embed torch.Size([10, 768])
    choices shape [torch.Size([5, 768]), torch.Size([5, 768]), torch.Size([5, 768])]
    album_title torch.Size([10, 8, 768])
    album_description torch.Size([21, 8, 768])
    album_when torch.Size([6, 8, 768])
    album_where torch.Size([768, 8])
    image_feats torch.Size([65, 2537])
    photo_titles [torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 
    768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), 
    torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768]), torch.Size([13, 768])]
    image_lengths torch.Size([8])
    
    """