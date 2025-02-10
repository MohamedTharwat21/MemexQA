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
from preprocess import (sentence2vec , strip_tags)



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
        self.valids = None    #"path to validation id list, if not set will be random 20%% of the training set"
        self.word_based = True   #"Word-based Embedding"



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
                                       self.cfg.word_based)
        answer_embed = sentence2vec(qa['answer'],
                                     self.cfg.word_based)
        
        # Process choices
        choices = qa['multiple_choices_4'][:]
        yidx = choices.index(qa['answer'])
        choices.remove(qa['answer'])
        choices_embed = [sentence2vec(c, self.cfg.word_based) for c in choices]

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
            

            # Get all tensors for photo titles
            photo_title_tensors = [sentence2vec(title, self.cfg.word_based) for title in album['photo_titles']]
            # Find the maximum length among the tensors
            max_len = max(tensor.size(0) for tensor in photo_title_tensors)
            # Pad tensors to the maximum length
            photo_title_tensors_padded = [
                torch.cat([tensor, torch.zeros(max_len - tensor.size(0), tensor.size(1))]) if tensor.size(0) < max_len else tensor
                for tensor in photo_title_tensors
            ]

            # Stack the padded tensors
            photo_titles.append(torch.stack(photo_title_tensors_padded))


            # pid -- photo id
            album_image_feats = [self.images[pid] for pid in album['photo_ids']]
            image_feats.append(np.stack(album_image_feats))
            image_lengths.append(len(album_image_feats))
        

        # Convert lists to tensors  
        def pad_and_stack(tensors, fixed_dim=None):
            """
            Pads a list of tensors along the first dimension and stacks them.

            Args:
                tensors (list): List of tensors to be padded and stacked.
                fixed_dim (int, optional): Fixed size for the second dimension. 
                                        If None, uses the size of the first tensor.

            Returns:
                Tensor: A single tensor with padded and stacked inputs.
            """
            # Ensure all tensors have at least 2 dimensions
            tensors = [tensor.unsqueeze(0) if tensor.ndim == 1 else tensor for tensor in tensors]

            # Find the maximum size for the first dimension
            max_dim0 = max(tensor.size(0) for tensor in tensors)

            # Use the size of the first tensor or the fixed dimension for the second dimension
            fixed_dim = fixed_dim if fixed_dim is not None else tensors[0].size(1)

            # Pad each tensor to the maximum size
            padded_tensors = [
                torch.cat([tensor, torch.zeros(max_dim0 - tensor.size(0), fixed_dim)], dim=0) if tensor.size(0) < max_dim0 else tensor
                for tensor in tensors
            ]

            # Stack all tensors
            return torch.stack(padded_tensors)

         

        print(f'the length of the album ids used in this question is {len(albumIds)}')
        # Choose between torch.stack and pad_and_stack
        if len(albumIds) == 1:
            # Directly stack since there’s only one album
            album_title = torch.stack(album_title)
            album_description = torch.stack(album_description)
            album_where = torch.stack(album_where)
            album_when = torch.stack(album_when)


        else:
            # Use pad_and_stack for multiple albums
            album_title = pad_and_stack(album_title)
            album_description = pad_and_stack(album_description)
            album_where = pad_and_stack(album_where)
            album_when = pad_and_stack(album_when)

        



        """ the length of the album ids used in this question is 7
            photos error
            torch.Size([9, 5, 768]) --> this is the first album of the 7's albums accessed by this question
            torch.Size([10, 9, 768])
            torch.Size([10, 9, 768])
            torch.Size([9, 7, 768])
            torch.Size([8, 4, 768])
            torch.Size([9, 6, 768])
            torch.Size([9, 6, 768])"""

         

        # Find maximum sizes for the first 2 dimensions
        max_dim1 = max(tensor.size(0) for tensor in photo_titles)  # max size along dim 0
        max_dim2 = max(tensor.size(1) for tensor in photo_titles)  # max size along dim 1
       
        # Pad each tensor to match the maximum sizes in dim 1 and dim 2
        def pad_tensor(tensor, max_dim1, max_dim2):
            # Create a tensor with the target size
            pad_shape = (max_dim1, max_dim2 , tensor.size(2))
            padded_tensor = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
            
            # Copy original tensor data into the padded tensor
            # the tensor will be add to it's fitting place .
            padded_tensor[:tensor.size(0), :tensor.size(1), :tensor.size(2)] = tensor
            return padded_tensor

        # Apply padding to all tensors
        padded_photo_titles = [pad_tensor(tensor, max_dim1, max_dim2) for tensor in photo_titles]
        
        # print("photos error fixing ")
        # for pht in padded_photo_titles:
        #     print(pht.shape)
            # """ torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
            #     torch.Size([10, 9, 768])
                
                # """


        # Concatenate along dimension 0
        photo_titles = torch.cat(padded_photo_titles, dim=0)
        # print(f"Resulting shape: {photo_titles.shape}") #torch.Size([70, 9, 768])

 






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







# from torch.utils.data import DataLoader

# Arguments and data loading
# cfg = ...  # Your arguments and pre-loaded data sources
# train_ids = [...]  # List of question IDs for training

# Initialize Dataset and DataLoader
# train_dataset = TrainDataset(cfg, train_ids)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)



# Training Loop
# for epoch in range(num_epochs):
#     for batch in train_loader:
#         question_embed, answer_embed, choices_embed, \
#         album_title, album_description, album_when, album_where, \
#         image_feats, photo_titles, image_lengths = batch
        
        # Perform training steps...






"""This setup ensures that you can work with 
very large datasets without running out of memory
or requiring preprocessing to fit everything in RAM."""




# The performance of high num_workers depends on the batch size
# and your machine. A general place to start is to set num_workers
# equal to the number of CPU cores on that machine. You can get the 
# number of CPU cores in Python using os.cpu_count() == i has 8 cores ,
# but note that depending on your batch size, you may overflow CPU RAM.