# DaSLaM

This repository contains code for the paper **Small Language Models Fine-tuned to Coordinate Larger Language Models improve Complex Reasoning**, accepted at **EMNLP Main** Conference 2023.

## 🛠 Dependencies and Installation
- torch `2.0.1+cu117`
- other dependencies in requirements.txt

  
## Supervised Finetune Stage
To finetune the model for question generation using SFT, 
Now run,
```python
python3 QuesGenFinetune.py
```

## RL Finetuning Stage
To further finetune the model using RLMF, make the following changes in file ppo_train_13B.py:

line 216: Replace the folder name with location of the 13B base llama model
line 223: Replace the folder name with location of the finetuned adapter
line 250: Replace the folder name with location of the 13B llama tokenizer 
line 263: Replace the folder name with location of the 13B base llama model
line 264: Replace the folder name with location of the 13B instruction finetuned llama adapter
line 266: Replace the folder name with location of the 13B llama tokenizer 

Now run,
```python
python3 LLAMA13B/Context/ppo_train_13B.py
```
