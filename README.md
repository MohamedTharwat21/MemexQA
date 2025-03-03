# MemexQA
MemexQA is a cutting-edge project designed to tackle the challenge of real-life multimodal question answering by leveraging both visual and textual data from personal photo albums. The task involves answering user queries about events or moments captured in sequences of photos, aiming to help users recover memories tied to these events. The model not only generates textual answers but also provides grounding photos as visual justifications, enabling users to quickly verify the provided answers.

Introduction

Data Characteristics

Problem Formulation

Modifications 
torch instead tensorflow
self-attention instead lstm--acc

Model Architecture

FOCAL VISUAL-TEXT ATTENTION (FVTA)

Installation
model params
venv
argparser









![image](https://github.com/user-attachments/assets/84ad409d-8205-455a-9def-11d2b7ba32ea)

Example FVTA Output.

![image](https://github.com/user-attachments/assets/799a77ea-1036-4145-8a92-7776385deed8)

Focal Visual-Text Attention Network. (old)

![image](https://github.com/user-attachments/assets/a11cff51-d16f-4027-aa7f-1dbc30040910)

Focal Visual-Text Attention Network. (old)

![image](https://github.com/user-attachments/assets/a0467903-0b91-44ee-8dd5-dc2af862e1b8)

## Dataset

![image](https://github.com/user-attachments/assets/c7196388-820b-460a-b632-a610f64a6808)
**Figure : Textual metadata, photos, question and four-choice answers.**

MemexQA dataset is composed of real-world personal photo albums and questions about events
captured in these photo sets. These photos capture events of a user’s life, such as a journey, a
ceremony, a part, etc. The questions are about these events and are designed to help a user to recall
the memory of these events.
MemexQA dataset has around 20K questions and answers on 5K personal photos with textual
metadata, including album title, album description, location, timestamp and photo titles. Fig2 shows
an example of an album. There are five types of questions in the dataset, including ‘When’, ‘What’,
‘Who’, ‘How many’, ‘Where’, whose distribution is showed in Fig3. Each question can be about a
single album or multiple albums; each question is provided with 4 candidate choices and one correct
answer out of them.

![image](https://github.com/user-attachments/assets/b9444dd5-efb8-4b73-b30f-6c238f3b3d99)
**Figure : Question distribution by question types.**




Model (new)

![image](https://github.com/user-attachments/assets/c3dacbce-bff5-4af9-8925-f5aadfa238e4)

MemexQA examples. The inputs are a question and a sequence
of a user’s photos with corresponding metadata. The outputs include a
short text answer and a few grounding photos to justify the answer.

![image](https://github.com/user-attachments/assets/a4e97ef4-d31b-49de-ae33-907668cbc67a)

Comparison of FVTA and classical VQA attention mechanism.
FVTA considers both visual-text intra-sequence correlations and cross
sequence interaction, and focuses on a few, small regions. In FVTA, the
multi-modal feature representation in the sequence data is preserved
without losing information.


## Baselines

### BERT-SL + Self-Attention + K/V Attention : 
To reduce the required training time and the too limited storage and GPU ram size, 
I directly utilize the sentence embeddings of text sequences from the pretrained BERT model as inputs. 
So, here, I don’t need to train the sequence encoder. Also, questions are encoded using the pooled
outputs from the pretrained BERT model. Also, to further reduce the required training time, i use
self-attention here to encode the information from different modalities, which are texts and images
here. Afterwards, I do the **key/value attention** between the questions, contexts, answers to obtain
the final attention map.

### BERT-WL + Self-Attention + K/V Attention :
Since sentence-level embeddings don’t seem to
have promising results, i switch to use word-level embebeddings from the pretrained BERT model
by sacrificing the required training time. In this setting, the sentence encoder is an simple **LSTM**.

### BERT-WL + FVTA **MemexQA_FVTA**: 
The origninal model which is **MemexQA** only uses the last hidden state of the encoder context,
which wastes a lot of information stored in the whole time hidden state. To get full use of the sequence
information, I implemented FVTA , which calculates the intra-sequence temporal
dependency between time step of each kind of data sequence, and cross sequence interaction between
different kind of data sequence and question sequence. With intra-sequence attention, the most
relevant time region of one data sequence to the question is found. With cross-sequence attention, the
most relevant data sequence to the question is found. 
Since i had some success with BERT-WL, i thought of doing the same to the GloVe-WL + FVTA model. As
i did before, I replaced GloVe with the pretrained BERT word embeddings of 768 dimensions.
Compared to Glove + FVTA, there was an accuracy improvement with BERT.


