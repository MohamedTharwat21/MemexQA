# MemexQA
MemexQA is a cutting-edge project designed to tackle the challenge of real-life multimodal question answering by leveraging both visual and textual data from personal photo albums. The task involves answering user queries about events or moments captured in sequences of photos, aiming to help users recover memories tied to these events. The model not only generates textual answers but also provides grounding photos as visual justifications, enabling users to quickly verify the provided answers.

![image](https://github.com/user-attachments/assets/b893d703-95bb-4a33-bb7e-586f1c7fb93d)


Data Characteristics

Problem Formulation

Modifications 
torch instead tensorflow
self-attention instead lstm--acc

Model Architecture

Installation
model params
venv
argparser









![image](https://github.com/user-attachments/assets/84ad409d-8205-455a-9def-11d2b7ba32ea)

Example FVTA Output.

 
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
**Figure : Question distribution by question types.**




## Model
![image](https://github.com/user-attachments/assets/c7196388-820b-460a-b632-a610f64a6808)
**Figure 3: The baseline model architecture.**

### Preprocessing
#### Visual Data
Images are embedded using a pre-trained Convolutional Neural Network (CNN) and further encoded with LSTM or self-attention.

#### Textual Data
Words in metadata, questions, and answers are embedded at both word-level (WL) and sentence-level (SL) before LSTM encoding.

#### Word-Level Embedding (WL)
Instead of using pre-trained GloVe embeddings, we experimented with BERT-based word embeddings (BERT-WL). As shown in Table 4.4.2, BERT-WL improves accuracy by 2.5% compared to GloVe-WL, achieving 54.8% accuracy.

#### Sentence-Level Embedding (SL)
We also experimented with using BERT-based sentence embeddings (BERT-SL), bypassing the sentence encoder to directly use pre-trained BERT embeddings as input.

### Modelling
#### Context Encoder
Different architectures are used to encode visual and textual sequences.

#### LSTM Encoder
Previous works used LSTM to encode contextual information but lacked temporal awareness for images and texts. To address this, we incorporated a self-attention mechanism.

#### Self-Attention Encoder
Since the input data lacks temporal structure, we exclude positional embeddings. Instead, image and text embeddings from the previous layer are treated independently.

![image](https://github.com/user-attachments/assets/3a59e1fd-d4dc-4c8d-adb5-5def631bc56f)
**Self-Attention mechanism**

### Attention Mechanism
#### Focal Visual-Text Attention (FVTA)
FVTA models the correlation between questions and multi-modal representations, emphasizing relevant context based on the question. Given that most answers are localized within short temporal segments, FVTA applies attention over focal context representations to summarize key information. A kernel tensor computes correlations between question words and context states, followed by a softmax layer to generate answer probabilities.

![image](https://github.com/user-attachments/assets/a4e97ef4-d31b-49de-ae33-907668cbc67a)
**FVTA considers both visual-text intra-sequence correlations and cross
sequence interaction, and focuses on a few, small regions. In FVTA, the
multi-modal feature representation in the sequence data is preserved
without losing information.**


#### Key/Value Attention (cross attention)
Unlike FVTA, this mechanism focuses only on final-step embeddings at either word or sentence level. Contextual embeddings are projected into key-value pairs, enabling attention over final representations to compute the attention map efficiently.

![image](https://github.com/user-attachments/assets/a13d7a3a-605e-452d-9d88-ff1b73cb5078)
**cross-attetnion**

## Baselines

### BERT-SL + Self-Attention + K/V Attention  
To reduce the required training time and the too limited storage and GPU RAM size,  
I directly utilize the sentence embeddings of text sequences from the pretrained BERT model as inputs.  
So, here, I don’t need to train the sequence encoder. Also, questions are encoded using the pooled  
outputs from the pretrained BERT model. To further reduce the required training time, I use  
self-attention to encode the information from different modalities, which are texts and images.  
Afterward, I perform **key/value attention** between the questions, contexts, and answers to obtain  
the final attention map.  

### BERT-WL + Self-Attention + K/V Attention  
Since sentence-level embeddings don’t seem to  
have promising results, I switch to using word-level embeddings from the pretrained BERT model  
by sacrificing the required training time. In this setting, the sentence encoder is a simple **LSTM**.  

### BERT-WL + FVTA **MemexQA_FVTA**  
The original model **MemexQA** only uses the last hidden state of the encoder context,  
which wastes a lot of information stored in the whole time hidden state. To fully utilize the sequence  
information, I implemented **FVTA**, which calculates the intra-sequence temporal  
dependency between the time step of each kind of data sequence and the cross-sequence interaction between  
different kinds of data sequences and the question sequence.  

- With **intra-sequence attention**, the most relevant time region of one data sequence to the question is found.  
- With **cross-sequence attention**, the most relevant data sequence to the question is found.  

Since I had some success with **BERT-WL**, I thought of doing the same with the **GloVe-WL + FVTA** model.  
As I did before, I replaced GloVe with the pretrained BERT word embeddings of 768 dimensions.  
Compared to **GloVe + FVTA**, there was an accuracy improvement with **BERT**.
