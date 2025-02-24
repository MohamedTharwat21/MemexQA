# MemexQA
MemexQA is a cutting-edge project designed to tackle the challenge of real-life multimodal question answering by leveraging both visual and textual data from personal photo albums. The task involves answering user queries about events or moments captured in sequences of photos, aiming to help users recover memories tied to these events. The model not only generates textual answers but also provides grounding photos as visual justifications, enabling users to quickly verify the provided answers.

Introduction

Data Characteristics

Problem Formulation

Model Architecture

FOCAL VISUAL-TEXT ATTENTION (FVTA)

Installation









![image](https://github.com/user-attachments/assets/84ad409d-8205-455a-9def-11d2b7ba32ea)

Example FVTA Output.

![image](https://github.com/user-attachments/assets/799a77ea-1036-4145-8a92-7776385deed8)

Focal Visual-Text Attention Network. (old)

![image](https://github.com/user-attachments/assets/a11cff51-d16f-4027-aa7f-1dbc30040910)

Focal Visual-Text Attention Network. (old)

![image](https://github.com/user-attachments/assets/a0467903-0b91-44ee-8dd5-dc2af862e1b8)

Dataset

![image](https://github.com/user-attachments/assets/c7196388-820b-460a-b632-a610f64a6808)

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
