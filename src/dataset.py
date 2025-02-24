"""
Key Changes :
--Code "preprocess.py" : No Pickling
    You no longer save the dataset to pickle files.
    Instead, you ensure that the data is prepared in 
    a structure that Code B can access dynamically.

--Code "dataset.py": Batch Loading
    Modify the TrainDataset class and other classes to load data
    from the source (e.g., JSON or raw files) on demand.
    This means that instead of loading the entire dataset into memory,
    only the data required for a batch is processed at a time.

--i will replace cfg with a class called cfg to contain the current
  configurations of my project
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from modified.preprocess import (sentence2vec , strip_tags)



class cfg:
    def __init__(self):
        self.albumjson=r'E:\Memex_QA\memexqa_dataset_v1.1\album_info.json'   #"path to album_info.json"
        self.qas=r'E:\Memex_QA\memexqa_dataset_v1.1\qas.json' #"path to the qas.json"
        self.glove='memexqa_dataset_v1.1/glove.6B.100d.txt'  #
        #"/path/to img feat npz file"
        self.imgfeat=r'E:\Memex_QA\memexqa_dataset_v1.1\photos_inception_resnet_v2_l2norm.npz'
        self.outpath=r'E:\Memex_QA\prepro'   #"output path"
        self.testids=r'E:\Memex_QA\memexqa_dataset_v1.1\test_question.ids'    # "path to test id list"
        self.use_BERT=True
        self.valids = None       #"path to validation id list, if not set will be random 20%% of the training set"
        self.word_based = False   #"Word-based Embedding"



"""
These all the data i want from the QAs 

	data = {
		'questions':questions,
		'questions_embed':questions_embed,   #will be returned 
		'answers':answers,                   #will be returned 
		'yidx': yidx,
		'aid':aid,
		'qid':qid,
		'choices':choices                    #will be returned 
	} 

    

These all the data i want from the Albums data

    album['album_id']	
    album['title']                           #will be returned 
    album['description']                     #will be returned 
    album['where']                           #will be returned 
    album['when']                            #will be returned 

    # photo info
    album['photo_urls'] 
    album['photo_titles']                    #will be returned 
    album['photo_ids'] 

    
and this is the last data i want (image features)
pid2feat['photo_id']

"""


class TrainDataset(Dataset):
    def __init__(self, cfg , question_ids):
        super(TrainDataset, self).__init__()
        
        # load data files 
        qas = json.load(open(cfg.qas,"r"))
        albums = json.load(open(cfg.albumjson,"r"))
        images = np.load(cfg.imgfeat)


        # Map all question IDs to their corresponding QA data
        self.qas = {str(qa['question_id']): qa for qa in qas}
        self.albums = {str(album['album_id']): album for album in albums}
        self.images = images  # Preloaded image features using pretained CNN (inception resnet v2 or other)
        self.question_ids = question_ids
        self.qid = []
        self.num_examples = len(self.question_ids)
        self.cfg = cfg  # Additional arguments
    


    # get just one training example , if batch = 8 then get just 8 so the overhead will be reduced on the ram size
    def __getitem__(self, idx):
                                 # train ids
        question_id = self.question_ids[idx]
        
        qa = self.qas[question_id] #qas data
        question = qa['question']         
        # self.qid.append(question_id)

        
        # Process question and answer
        question_embed = sentence2vec(qa['question'],
                                       True)
        answer_embed = sentence2vec(qa['answer'],
                                     True)
        
        # Process choices
        choices = qa['multiple_choices_4'][:]
        yidx = choices.index(qa['answer'])
        choices.remove(qa['answer'])
        choices_embed = [sentence2vec(c, True) for c in choices]

        # Process album data

        """here i will take a qas example , then looking at album_ids inside it ,
           then bringing just the album data i want now for this example 
           , not as i did in preprocess.py i were loading the whole qas then take the global_ids for
           all the used albums , then making another loop to iterate on the whole albums to save their data
           after saving qas data . """
        
        albumIds = [str(album_id) for album_id in qa['album_ids']] #this line would be loaded from the pickle 
        album_title, album_description = [], []
        album_where, album_when = [], []
        photo_titles, image_feats, image_lengths = [], [], []
        
        # iterating on the albumIds just are being used now in the current example
        for albumId in albumIds:
            album = self.albums[albumId]
            album_title.append(sentence2vec(album['album_title'],
                                             self.cfg.word_based))
            album_description.append(sentence2vec(strip_tags(album['album_description']),
                                                  self.cfg.word_based))
            
            if album['album_where'] is None:
                album_where.append(torch.zeros(768))
            else:
                album_where.append(sentence2vec(album['album_where'],
                                                self.cfg.word_based))
            
            album_when.append(sentence2vec(album['album_when'],
                                            self.cfg.word_based))
            
            # Photo Titles
            # photo_titles.append(torch.stack([sentence2vec(title, \
            #         self.cfg.word_based) for title in album['photo_titles']]))
            
            # photo titles 
            # Get all tensors for photo titles
            photo_title_tensors = [sentence2vec(title, self.cfg.word_based) for title in album['photo_titles']]
            photo_titles.append(torch.stack(photo_title_tensors))

            # images 
            # pid -- photo id
            album_image_feats = [self.images[pid] for pid in album['photo_ids']]
            image_feats.append(np.stack(album_image_feats))
            image_lengths.append(len(album_image_feats))


        # print(f'the length of the album ids used in this question is {len(albumIds)}')


        album_title = torch.stack(album_title)
        # album_title = pad_sequence(album_title, batch_first=True)

        album_description = torch.stack(album_description)
        # album_description = pad_sequence(album_description, batch_first=True)

        album_when = torch.stack(album_when)
        # album_when = pad_sequence(album_when, batch_first=True)

        album_where = torch.stack(album_where)
        # album_where = pad_sequence(album_where, batch_first=True)

        # photo titles
        photo_titles = torch.cat(photo_titles, 0)

        # images
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))
        image_lengths = torch.LongTensor(image_lengths)
        
        """return [the raw question,ans embedding,the remaining 3 choices embedding after excluding the answer];
           return data for albums -embedded- [title,description,when,where];
           return data for the images/photos in the current album [numberOfimages in each album,features for each images of them,and their titles]"""
        return question_embed.detach(),\
               answer_embed.detach(),  \
               [cho.detach() for cho in choices_embed], \
               album_title.detach(), \
               album_description.detach(), \
               album_when.detach(), \
               album_where.detach(), \
               image_feats.detach(), \
               photo_titles.detach(), \
               image_lengths.detach()



    def __len__(self):
        return self.num_examples
    

class DevDataset(Dataset):
    def __init__(self, cfg , question_ids):
        super(DevDataset, self).__init__()
        
        # load data files 
        qas = json.load(open(cfg.qas,"r"))
        albums = json.load(open(cfg.albumjson,"r"))
        images = np.load(cfg.imgfeat)


        # Map all question IDs to their corresponding QA data
        self.qas = {str(qa['question_id']): qa for qa in qas}
        self.albums = {str(album['album_id']): album for album in albums}
        self.images = images  # Preloaded image features using pretained CNN (inception resnet v2 or other)
        self.question_ids = question_ids
        self.qid = []
        self.num_examples = len(self.question_ids)
        self.cfg = cfg  # Additional arguments
    
    def __getitem__(self, idx):
                                   # val ids
        question_id = self.question_ids[idx]    
        qa = self.qas[question_id] #qas data
        question = qa['question']         
        # Process question and answer
        question_embed = sentence2vec(qa['question'],
                                       True)
        answer_embed = sentence2vec(qa['answer'],
                                     True)
        
        # Process choices
        choices = qa['multiple_choices_4'][:]
        yidx = choices.index(qa['answer'])
        choices.remove(qa['answer'])
        choices_embed = [sentence2vec(c, True) for c in choices]
        albumIds = [str(album_id) for album_id in qa['album_ids']] #this line would be loaded from the pickle 
        album_title, album_description = [], []
        album_where, album_when = [], []
        photo_titles, image_feats, image_lengths = [], [], []
        
        # iterating on the albumIds just are being used now in the current example
        for albumId in albumIds:
            album = self.albums[albumId]
            album_title.append(sentence2vec(album['album_title'],
                                             self.cfg.word_based))
            album_description.append(sentence2vec(strip_tags(album['album_description']),
                                                  self.cfg.word_based))
            
            if album['album_where'] is None:
                album_where.append(torch.zeros(768))
            else:
                album_where.append(sentence2vec(album['album_where'],
                                                self.cfg.word_based))
            
            album_when.append(sentence2vec(album['album_when'],
                                            self.cfg.word_based))
            # photo titles 
            # Get all tensors for photo titles
            photo_title_tensors = [sentence2vec(title, self.cfg.word_based) for title in album['photo_titles']]
            photo_titles.append(torch.stack(photo_title_tensors))

            # images 
            # pid -- photo id
            album_image_feats = [self.images[pid] for pid in album['photo_ids']]
            image_feats.append(np.stack(album_image_feats))
            image_lengths.append(len(album_image_feats))

        album_title = torch.stack(album_title)
        album_description = torch.stack(album_description)
        album_when = torch.stack(album_when)
        album_where = torch.stack(album_where)
        # photo titles
        photo_titles = torch.cat(photo_titles, 0)
        # images
        image_feats = torch.Tensor(np.concatenate(image_feats, 0))
        image_lengths = torch.LongTensor(image_lengths)
        
        return question_embed.detach(),\
               answer_embed.detach(),  \
               [cho.detach() for cho in choices_embed], \
               album_title.detach(), \
               album_description.detach(), \
               album_when.detach(), \
               album_where.detach(), \
               image_feats.detach(), \
               photo_titles.detach(), \
               image_lengths.detach()



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



def get_ids():
    import random 
    manualSeed = 1126
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Load train, val and test ids
    json_file=  open(r'modified\IDs.json', 'r')  
    loaded_data = json.load(json_file)

    # print(len(loaded_data['trainIds']))
    train_ids , val_ids , test_ids = loaded_data['trainIds'] , loaded_data['valIds'] , \
                                    loaded_data['testIds']
    
    return train_ids , val_ids , test_ids



def loader_data(cfg):
    from torch.utils.data import DataLoader
    from single_album.test_collate_fn import collate_fn
    # path_to_pickle = r'E:\Memex_QA\single_album\prepro'

    train_ids , val_ids , test_ids = get_ids()

    train_dataset = TrainDataset(cfg , train_ids)

    print(f'length of train_dataset {len(train_dataset)}')
   
    # question_embed, answer, choice, \
    # album_title, album_description, album_when, \
    # album_where, image_feats, \
    # photo_titles, image_lengths = train_dataset[15]
    
    train = DataLoader(train_dataset, batch_size = 8, 
                       shuffle=True, num_workers=0,
                       collate_fn=collate_fn) 
    return train


def test_model(train):
    from single_album.model import MemexQA , MemexQA_FVTA
    device = torch.device("cpu")
    
    # model = MemexQA(input_dim=768,  #word embedding of bert
    #             img_dim=2537,      #features of inception-resnet-v2
    #             hidden_dim=768,
    #             key_dim=32, #this is the dimension after div by num_heads 
    #             value_dim=32, 
    #             num_label=2, 
    #             num_head=4,
    #             num_layer=2, 
    #             mode="one-shot", #one-shot
    #             device=device) #cuda

    model = MemexQA_FVTA(input_dim=768,  #word embedding of bert
                img_dim=2537,      #features of inception-resnet-v2
                hidden_dim=768,
                key_dim=32, #this is the dimension after div by num_heads 
                value_dim=32, 
                num_label=2, 
                num_head=4,
                num_layer=2, 
                mode="one-shot", #one-shot
                device=device) #cuda
    

    
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



if __name__ == "__main__":
    
    # Load data and print out the first item in the batch
         
    cfg = cfg()
    train = loader_data(cfg=cfg)

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
