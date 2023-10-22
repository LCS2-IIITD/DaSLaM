import torch
import nltk
import signal
from io import StringIO
import re 
import spacy
import time
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

nlp = spacy.load('en_core_web_sm')

import sys
import contextlib


def reward1(input):
    # breakpoint()
    reward = []
    for i in range(len(input)):
        question = input[i][1]
        sub_question = input[i][2]
        nouns_question = get_nouns([question])
        nouns_sub_question = get_nouns(sub_question)
        try:
            difference = 1 - len(nouns_question.symmetric_difference(nouns_sub_question))/len(nouns_question)
        except:
            difference = 1
        reward.append(torch.tensor(difference))
    return reward

def get_nouns(sentence):
    # print("INSIDE GET NOUNS")
    # breakpoint()
    nouns = []
    for s in sentence:
        tagged_words = nltk.pos_tag(nltk.word_tokenize(s))
        for word, pos in tagged_words:
            if pos.startswith('NN'):
                nouns.append(word)
    return set(nouns)

def reward2_cot(questions, outputs):
    s = time.time()
    r=torch.tensor(0.0)
    # breakpoint()
    for i in range(len(questions)):
        # Get the word embeddings for each sentence
        embedding1 = nlp(questions[i]).vector
        embedding2 = nlp(outputs[i]).vector

        # Calculate the cosine similarity between the embeddings
        similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
        r+= torch.tensor(0.25*similarity_score)
    e = time.time()
    # print("TIME FOR REWARD 2", e-s)
    return r

def reward3_cot(answers, outputs, dones_prev, reward1, reward2):
    reward=[]
    dones = []
    # breakpoint()
    s = time.time()
    for i in range(len(answers)):
        outputs = find_answer(outputs)
        print("#### OUTPUTS:", outputs[i], answers[i])
        try:
            answers[i] = answers[i].strip()
        except:
            pass
        try:
            outputs[i] = outputs[i].strip()
        except:
            pass


        if(outputs[i] == answers[i]):
            print("REWARD +100")
            r = torch.tensor(1.0)
            dones.append(True)
        else:
            r= torch.tensor(0.0)
            dones.append(False)
            # print(outputs[i], answers[i])
        reward.append(r + reward1[i] + reward2[i])
        
    e = time.time()
    # print("TIME FOR REWARD 3", e-s)
    return dones, reward

def find_answer(outputs):
    answers = []
    # breakpoint()
    for output in outputs:
        try:
            pattern = re.compile(re.escape("###"))
            match = pattern.search(output)
            if(match):
                output = output[:match.start()]
                
            matches = re.findall(r'\[\[.*?\]\]', output)
            for match in matches:
                # output = re.sub(r'[^]', '', match)
                output = match[9:-2]
                output = output.strip()
                # output = output[-1]
                
            
        except:
            # print("##### PROBLEMATIC OUTPUT", output)
            pass
    
        answers.append(output)
    return answers

def reward4_cot(Ans1, Ans2, Ans3):
    # breakpoint()
    reward_list = []
    for i in range(len(Ans1)):
        cot1 = Ans1[i]
        cot2 = Ans2[i]
        cot3 = Ans3[i]
        reward = torch.tensor(0.0)
        cot1_list = cot1.split("\n")
        cot2_list = cot2.split("\n")
        cot3_list = cot3.split("\n")
        try:
            for i in range(min(min(len(cot1_list), len(cot2_list)), len(cot3_list))):
                cosine1 = cosine_similarity([nlp(cot1_list[i]).vector], [nlp(cot2_list[i]).vector])[0][0]
                cosine2 = cosine_similarity([nlp(cot1_list[i]).vector], [nlp(cot3_list[i]).vector])[0][0]
                if(cosine2 > cosine1):
                    reward += torch.tensor(0.5*cosine1)
                else:
                    reward += -1 - torch.tensor(0.5*cosine1)
        except:
            reward = torch.tensor(0.0)
        reward_list.append(reward)
    return reward_list
        
def find_operations(str1, operations):
    operations_found = []
    
    for operation in operations:
        if operation in str1:
            operations_found.append(operation)
    
    return operations_found

operations_list = ['+', '-', '*', '/', 'sin', 'log', '^', 'cos', 'tan', 'cot', 'sec', 'cosec', 'sqrt', 'pi', 'exp']

def reward5_cot(Ans1, Ans2):
    # breakpoint()
    rewards = []
    for i in range(len(Ans1)):
        cot1 = Ans1[i]
        cot2 = Ans2[i]
        reward = torch.tensor(0.0)
        operations_list_1 = find_operations(cot1, operations_list)
        operations_list_2 = find_operations(cot2, operations_list)
        for i in range(min(len(operations_list_1), len(operations_list_2))):
            if(operations_list_1[i] == operations_list_2[i]):
                reward += torch.tensor(0.5)
            else:
                reward += torch.tensor(-0.5)
        rewards.append(reward)
    return rewards

        



