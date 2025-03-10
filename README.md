# MemexQA
MemexQA is a cutting-edge project designed to tackle the challenge of real-life multimodal question answering by leveraging both visual and textual data from personal photo albums. The task involves answering user queries about events or moments captured in sequences of photos, aiming to help users recover memories tied to these events. The model not only generates textual answers but also provides grounding photos as visual justifications, enabling users to quickly verify the provided answers.

![image](https://github.com/user-attachments/assets/0b120b6b-6ecb-4638-8a17-a0a52ea5ced8)

## Contents
0. [Dataset](#dataset)
0. [Model](#model)
0. [Baselines](#baselines)
0. [FVTA](#fvta)
0. [Memex Design on lucidchart](#Memex-Design-on-Lucidchart)
0. [Label Smoothing Loss](#Label-Smoothing-Loss)
0. [Running the Pipeline (Linux & Windows)](#running-the-pipeline-linux--windows)
0. [Installation](#installation)
0. [Parameters](#parameters)

## Dataset

![image](https://github.com/user-attachments/assets/e440360e-b5ca-4e9c-922c-00ebefbafcd1)

**Figure 2 : Textual metadata, photos, question and four-choice answers.**

MemexQA dataset is composed of real-world personal photo albums and questions about events
captured in these photo sets. These photos capture events of a user’s life, such as a journey, a
ceremony, a part, etc. The questions are about these events and are designed to help a user to recall
the memory of these events.

MemexQA dataset has around **20K questions and answers on 5K personal photos with textual
metadata, including album title, album description, location, timestamp and photo titles** . 
Fig2 shows an example of an album. There are five types of questions in the dataset, including ‘When’, ‘What’,
‘Who’, ‘How many’, ‘Where’, whose distribution is showed in Fig3. Each question can be about a
single album or multiple albums; each question is provided with 4 candidate choices and one correct
answer out of them.


![image](https://github.com/user-attachments/assets/c3dacbce-bff5-4af9-8925-f5aadfa238e4)

**MemexQA examples. The inputs are a question and a sequence
of a user’s photos with corresponding metadata. The outputs include a
short text answer and a few grounding photos to justify the answer.**


![image](https://github.com/user-attachments/assets/b9444dd5-efb8-4b73-b30f-6c238f3b3d99)

**Figure 3 : Question distribution by question types.**

# Dataset download
	memexqa_dataset_v1.1/
	├── album_info.json   # album data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
	├── glove.6B.100d.txt # word vectors for baselines:  http://nlp.stanford.edu/data/glove.6B.zip
	├── photos_inception_resnet_v2_l2norm.npz # photo features: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
	├── qas.json # QA data: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
	└── test_question.ids # testset Id: https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids

# Collect Dataset
    mkdir memexqa_dataset_v1.1 
    cd memexqa_dataset_v1.1 
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/album_info.json
    wget http://nlp.stanford.edu/data/glove.6B.zip
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/qas.json
    wget https://memexqa.cs.cmu.edu/memexqa_dataset_v1.1/test_question.ids
    unzip glove.6B.zip


## Model
![image](https://github.com/user-attachments/assets/c7196388-820b-460a-b632-a610f64a6808)

**Figure 4 : The baseline model architecture.**

### Preprocessing
#### Visual Data
Images are embedded using a pre-trained Convolutional Neural Network (CNN) and further encoded with LSTM or self-attention.
#### Textual Data
Words in metadata, questions, and answers are embedded at both word-level (WL) and sentence-level (SL) before LSTM encoding.
#### Word-Level Embedding (WL)
Instead of using pre-trained GloVe embeddings, i experimented with BERT-based word embeddings (BERT-WL). 
#### Sentence-Level Embedding (SL)
I also experimented with using BERT-based sentence embeddings (BERT-SL), bypassing the sentence encoder to directly use pre-trained BERT embeddings as input.

### Modelling
#### Context Encoder
I used several different architectures to encode visual and text sequences respectively. In current
experiments, i use **Self-Attention** and **LSTM** as encoder. The inputs are **image/text** embedding
produced by the previous layer. The outputs of each encoder network at each step is concatenated as
a representation of the image, metadata and text sequences.
#### LSTM Encoder
To encode a series of information, the original work proposed by ["Focal Visual-Text Attention for Memex Question Answering"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8603827) used an LSTM to encode
all contextual information. A drawback with this approach is that they don’t have **any temporal
information for the images and texts** . So, i revise this part with the **self-attention architecture**.

#### Self-Attention Encoder
Since the input data lacks temporal structure, i exclude positional embeddings. Instead, image and text embeddings from the previous layer are treated independently.
![image](https://miro.medium.com/v2/resize:fit:1400/1*7Jbg-m9UNKXw-Mw_o-9BJQ.gif)

**Figure 5 :Self-Attention mechanism**

### Attention Mechanism
#### Focal Visual-Text Attention (FVTA)
FVTA models the correlation between questions and multi-modal representations, emphasizing relevant context based on the question. Given that most answers are localized within short temporal segments, FVTA applies attention over focal context representations to summarize key information. A kernel tensor computes correlations between question words and context states, followed by a softmax layer to generate answer probabilities.

![image](https://github.com/user-attachments/assets/9fef9bea-3c39-43c5-baf9-965860107431)

**Figure 6 :FVTA considers both visual-text intra-sequence correlations and cross
sequence interaction, and focuses on a few, small regions. In FVTA, the
multi-modal feature representation in the sequence data is preserved
without losing information.**


#### Key/Value Attention (cross attention)
Unlike FVTA, this mechanism focuses only on final-step embeddings at either word or sentence level. Contextual embeddings are projected into key-value pairs, enabling attention over final representations to compute the attention map efficiently.

![image](https://github.com/user-attachments/assets/a13d7a3a-605e-452d-9d88-ff1b73cb5078)

**Figure 7 :cross-attetnion**

## Baselines

### BERT-SL + Self-Attention + K/V Attention  
To reduce the required training time and the too limited storage and GPU RAM size, I directly utilize the sentence embeddings of text sequences from the pretrained BERT model as inputs. So, here, I don’t need to train the sequence encoder. Also, questions are encoded using the pooled outputs from the pretrained BERT model. To further reduce the required training time, I use self-attention to encode the information from different modalities, which are texts and images.Afterward, I perform **key/value attention** between the questions, contexts, and answers to obtain the final attention map.  
### BERT-WL + Self-Attention + K/V Attention  
Since sentence-level embeddings don’t seem to have promising results, I switch to using word-level embeddings from the pretrained BERT model by sacrificing the required training time. In this setting, the sentence encoder is a simple **LSTM**.  
### BERT-WL + FVTA **MemexQA_FVTA**  
The original model **MemexQA** only uses the last hidden state of the encoder context, which wastes a lot of information stored in the whole time hidden state. To fully utilize the sequence information, I implemented **FVTA**, which calculates the intra-sequence temporal dependency between the time step of each kind of data sequence and the cross-sequence interaction between different kinds of data sequences and the question sequence.  
- With **intra-sequence attention**, the most relevant time region of one data sequence to the question is found.  
- With **cross-sequence attention**, the most relevant data sequence to the question is found.  
Since I had some success with **BERT-WL**, I thought of doing the same with the **GloVe-WL + FVTA** model. As I did before, I replaced GloVe with the pretrained BERT word embeddings of 768 dimensions. Compared to **GloVe + FVTA**, there was an accuracy improvement with **BERT**.

## FVTA

![memexqa_simp](https://github.com/user-attachments/assets/6ae7ebbd-a8de-49d4-9c64-a822d2145f5d)

![image](https://github.com/user-attachments/assets/3debb4b1-e852-4320-b800-5288af1161a2)

**Figure 8 & 9 : An overview of Focal Visual-Text Attention (FVTA) model. For visual-text embedding, using pre-trained convolutional neural
network to embed the photos and pre-trained word vectors to embed the words. and using a bi-directional LSTM as the sequence encoder.
All hidden states from the question and the context are used to calculate the FVTA tensor. Based on the FVTA attention, both question and
the context are summarized into single vectors for the output layer to produce final answer. The output layer is used for multiple choice
question classification (I used four-choices). The text embedding of the answer choice is also used as the input. This input is not shown in the figure.**



## Memex Design on lucidchart


![image](https://github.com/user-attachments/assets/3c3cfc47-f468-4988-9d67-5af102a90bd0)
**Questions , Answers , Images , Text Encodings**


----------------------------------------
![image](https://github.com/user-attachments/assets/6ee95eb1-40d9-48b4-8785-ccb9fdbac52a)
**Images Multi-headed Self Attention**


----------------------------------------
![image](https://github.com/user-attachments/assets/6b4545a5-3177-4fe3-bd65-23737773a566)
**Texts Multi-headed Self Attention**


----------------------------------------
![image](https://github.com/user-attachments/assets/e0754173-c635-467c-b8c6-8441c3b439f2)
**Concatenation of both Texts and Images Multi-headed Self Attention Representations**


----------------------------------------
![image](https://github.com/user-attachments/assets/b838cb72-9dc3-423f-a41d-3f0ac47c5fa5)
**Passing Context Encoder to Multi-headed Self Attention Many Times and Finally passing it to Cross Attention OR Focal visual-text Attention**


## Running the Pipeline (Linux & Windows)

### Linux
Preprocess:
``` 
python3 src/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz prepro_BERT_SA_sb
``` 
Train:
```
python3 src/train.py --workers 8 --batchSize 32 --niter 100 --inpf ./prepro_BERT_SA_sb/ --outf ./outputs/BERT_WL_SA_sb --cuda --gpu_id 0
```
Test:
```
python3 src/test.py --workers 8 --batchSize 32 --inpf ./prepro_BERT_SA_sb/ --outf ./outputs/BERT_WL_SA_sb --cuda --gpu_id 0
```

### Windows
Preprocess:
```
python src/preprocess.py memexqa_dataset_v1.1/qas.json memexqa_dataset_v1.1/album_info.json memexqa_dataset_v1.1/test_question.ids memexqa_dataset_v1.1/photos_inception_resnet_v2_l2norm.npz prepro_BERT_SA_sb
```
Train:
```
python src/train.py --workers 8 --batchSize 32 --niter 100 --inpf ./prepro_BERT_SA_sb/ --outf ./outputs/BERT_WL_SA_sb --cuda --gpu_id 0
```
Test:
```
python src/test.py --workers 8 --batchSize 32 --inpf ./prepro_BERT_SA_sb/ --outf ./outputs/BERT_WL_SA_sb --cuda --gpu_id 0
```
 
 
## Installation
Follow these steps to set up the project on your machine.

### 1. Clone the Repository
First, download the project by cloning the repository:
```
git clone https://github.com/MohamedTharwat21/MemexQA.git
cd your-repo
```

### 2. Set Up a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.
For Linux/macOS:
```
python3 -m venv venv
source venv/bin/activate
```
For Windows:
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Once the virtual environment is activated, install the required packages:
```
pip install -r src/requirements.txt
```

### 4. Verify Installation
Check that the necessary dependencies are installed correctly:
```
python -m pip list
```

### 5. Run a Quick Test
To ensure everything is set up properly, try running:
```
python src/preprocess.py --help
```
