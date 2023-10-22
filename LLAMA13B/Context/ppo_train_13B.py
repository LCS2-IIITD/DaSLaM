# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from Environment import Reward
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json 
import os
import re
import sys
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import wandb
import warnings
import time
warnings.filterwarnings("ignore")

num_epochs = 1
device_model = 0
device_model_cuda = "cuda:0"
device_env = 0
device_env_cuda = "cuda:0"
max_len = 2048
batch_size = 8

def remove_common_prefix(i_str1, i_str2):
        r_str2=[]
        # # breakpoint()
        for i in range(len(i_str1)):
            str1=i_str1[i]
            str2=i_str2[i]
            # pattern = re.compile(re.escape(str1))
            # match = pattern.search(str2)
            # if(match):
            #     str2 = str2[match.end():]

            pattern = re.compile(re.escape("### Response:"))
            match = pattern.search(str2)
            if(match):
                str2 = str2[match.end():]

            pattern = re.compile(re.escape("###"))
            match = pattern.search(str2)
            if(match):
                str2 = str2[:match.end()-3]


            pattern = re.compile(re.escape("<unk>"))
            match = pattern.search(str2)
            if(match):
                str2 = str2[:match.end()-5]
            r_str2.append(str2)
            # print(str2)
        return r_str2

def extract_answer_math(sentence):
    # Example string
    string = sentence

    # Regular expression pattern to match content inside curly braces
    pattern = r"boxed\{(.*?)\}."

    # Find all matches of the pattern in the string
    matches = re.findall(pattern, string)
    # print(sentence)
    # Print the extracted contents
    for match in matches:
      
        # print(match)
        return match
    return ""
   
class StringDataset(Dataset):
    def __init__(self, qstrings, astrings, tag,cot):
        self.qstrings = qstrings
        self.astrings = astrings
        self.tag = tag
        self.cot = cot

    def __len__(self):
        return len(self.astrings)

    def __getitem__(self, idx):
        return self.qstrings[idx], self.astrings[idx], self.tag[idx], self.cot[idx]

class StringDataLoader(DataLoader):
    def __init__(self, qstrings, astrings,Tags, cot, batch_size=1, shuffle=False, num_workers=0):
        dataset = StringDataset( qstrings, astrings, Tags, cot)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def create_dataset_StrategyQA(filename, bs):
    # open dev.json file. this file has data in the format [{"question": "question", "answer": "answer", "facts":facts, "decomopsition": decomposition}]. make arrays of questions, answers, facts and decompositions
    Questions = []
    Answers=[]
    Facts = []
    Decomposition = []
    Tag = []
    with open(filename) as f:
        data = json.load(f)

    for item in data:
        question = item['question']
        answer = item['answer']
        facts = item['facts']
        fact_str = ""
        for fact in facts:
            fact_str += fact + "\n"
        decomposition = item['decomposition']
        dec = ""
        for decomp in decomposition:
            dec += decomp + '\n'

        Questions.append(question)
        Answers.append(answer)
        Facts.append(fact_str)
        Decomposition.append(dec)
        Tag.append("SQA")
    
    return Questions, Answers, Facts, Decomposition, Tag

def create_dataset_MATH(directory, bs):
    Questions = []
    Reasoning = []
    Answers = []
    Tag = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                if(extract_answer_math(data["solution"]).isdigit()):
                    # Extract problem and solution from the JSON data
                    Questions.append(data["problem"])
                    rs = data["solution"].split('.')
                    step = ""
                    for r in rs:
                        step+=r+'\n'
                    Reasoning.append(step[:-2])
                    Answers.append(extract_answer_math(data["solution"]))
                    Tag.append("MATH")


    return Questions, Answers, Reasoning, Tag

def create_dataset_AQuA(filename, bs):
    Questions = []
    Answers=[]
    Rationale = []
    Answers = []
    Tag = []
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        question = result['question']
        rs = result["rationale"]
        step = ""
        for r in rs:
            step+=r+'\n'
        Rationale.append(rs)
        # Answers.append(step)
        options = result['options']
        for option in options:
            question += " " + option + " "
        # Questions.append(result['question'])
        # Questions.append(result['problem'])
        Questions.append(question)
        # Answers.append(extract_answer(result['answer']))
        Answers.append(result['correct'])
        Tag.append("AQuA")
    return Questions, Answers, Rationale, Tag

### Copy the location of the dataset files / folders here
Q_stratergy, A_stratergy, COT_stratergy, SQ_stratergy, T_SQ = create_dataset_StrategyQA("./strategyqa/dev.json", 1)
Q_pnc, A_pnc, COT_pnc, T_pnc = create_dataset_MATH("./MATH/test/counting_and_probability", 1)
Q_NT, A_NT, COT_NT, T_NT = create_dataset_MATH("./MATH/test/number_theory", 1)
Q_ialg, A_ialg, COT_ialg, T_ialg = create_dataset_MATH("./MATH/test/intermediate_algebra", 1)
Q_alg, A_alg, COT_alg, T_alg = create_dataset_MATH("./MATH/test/algebra", 1)
Q_AQuA, A_AQuA, COT_AQuA, T_AQuA = create_dataset_AQuA("./AQuA/test.json", 1)

Q_geom, A_geom, COT_geom, T_geom = create_dataset_MATH("./MATH/test/geometry", "G")
Q_palg, A_palg, COT_palg, T_palg = create_dataset_MATH("./MATH/test/prealgebra", "PALG")
Q_cal, A_cal, COT_cal, T_cal = create_dataset_MATH("./MATH/test/precalculus", "PCAL")

Q =  Q_palg
A =  A_palg
T =  T_palg
COT = COT_palg

train_dataloader = StringDataLoader(Q, A, T, COT, batch_size = batch_size)

wandb.init(project="Reasoning_context", name = "GPT3", 
           config = {
               'init_kl_coef': 0.01, 
               'target': 4, 
               'gradient_accumulation_steps':8
           })

# initialize trainer
ppo_config = PPOConfig(batch_size=batch_size, gradient_accumulation_steps=8, init_kl_coef=0.01, target=4, optimize_cuda_cache=True, log_with="wandb")

# breakpoint()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

### Location of your pretrained model here
model = LlamaForCausalLM.from_pretrained(
    "./13B_HF/",
    load_in_8bit=True,
    device_map={"":device_model},
)

### Load the SFT Adapter here
model = PeftModel.from_pretrained(model, "./GPT_Finetuned")

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model,
    load_in_8bit=True,
    device_map={"": device_model},
)

for param in model.modules():
    for name, value in param.named_parameters():
        # if('lora' in name):
            # print(name)

        if "pretrained_model.base_model.model.model.layers.39" in name and "lora" in name:
            # print(name)
            value.requires_grad = True
        if "pretrained_model.base_model.model.model.layers.38" in name and "lora" in name:
            # print(name)
            value.requires_grad = True
        if "pretrained_model.base_model.model.model.layers.37" in name and "lora" in name:
            # print(name)
            value.requires_grad = True
        if "pretrained_model.base_model.model.model.layers.36" in name and "lora" in name:
            # print(name)
            value.requires_grad = True

### Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("./13B_HF/tokenizer.model", padding_side = "left")
tokenizer.pad_token_id = 0


generation_config = GenerationConfig(
    temperature=0.2,
    top_p = 0.18
)


ppo_trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=tokenizer)

### Load the environment model and tokenizer
model_env = LlamaForCausalLM.from_pretrained("./13B_HF/", load_in_8bit=True, device_map={"": device_env})
model_env = PeftModel.from_pretrained(model_env, "./lora-alpaca-13B")

tok_env = LlamaTokenizer.from_pretrained("./13B_HF/tokenizer.model", padding_side = "left")
tok_env.pad_token_id = 0


with open('./LLAMA13B/prompts/MATH.txt', 'r') as file:
        prompt_small = file.read()
with open('./LLAMA13B/prompts/MATH.txt', 'r') as file:
        prompt_large = file.read()

env = Reward(model_env, tok_env, batch_size= batch_size, prompt_small = prompt_small, prompt_large = prompt_large, device=device_env_cuda)


def respond(model, questions):

    input_ids = tokenizer(questions, return_tensors="pt", padding=True, truncation = True, max_length = 256).input_ids.to(device_model_cuda)
    outputs=model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=128,
        )
    result = []
    for s in outputs.sequences:
         result.append(tokenizer.decode(s))
    
    result = remove_common_prefix(questions, result)
    return result

r=[]
ste = 0 
acc1 = 0
acc2 = 0
acc3 = 0
x=0

def extract_questions(text):
    # Define the regular expression pattern for questions
    pattern = r"(?:^|\s)[\w\s\d\W]+?\?"

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)

    # Filter out any non-question matches
    questions = [match.strip() for match in matches if '?' in match]

    return questions

# breakpoint()
solved_0 = [0,0,0]
solved_1 = [0,0,0]
for _ in range(num_epochs):
    batch_number = 0
    for batch in tqdm(train_dataloader):
        x+=1
        reward_epoch = 0
        # main idea: for a batch, we will send all the questions that are not done on one pass

        questions, answers, tags, reasoning = batch
           
        done_all = False

        s = time.time()
        # we would first ask the environment to respond to the questions, based on that response, we will generate further sub questions
        dones_prev=[False]*batch_size
        reasoning_prev = ["I am "]*batch_size
        dones_0, outputs_0, rewards_0, cot_0 = env.step(True, [], questions, answers, dones_prev, reasoning, reasoning_prev)
        s1=time.time()
        for i in range(len(dones_0)):
            if(dones_0[i]):
                if(T[i]=="MATH"):
                    solved_0[0]+=1
                if(T[i]=="AQuA"):    
                    solved_0[1]+=1

        # print(dones_0[0], outputs[0], rewards_0[0])
        # print("GENERATED FIRST PASS", s1-s)
        # sys.exit()
        # break
        i=0
        outputs=[]
        # outputs=outputs_0
        # print("NOT DONE", i)
        # breakpoint()
        # outputs = question + code generated by environment
        # breakpoint()
        questions_n =[]
        answers_n = []
        tags_n = []
        reasoning_n = []
        cot_prev = []
        for j in range(len(questions)):
            if dones_0[j]==False:
                outputs.append(f'''
                Below is an instruction that describes a task, paired with an input and a reasoning that provides further context. Write a response that appropriately completes the request.

                    ### Instruction: Break the input question into multiple subquestions based on the reasoning provided. 
    
                    ### Input:  {questions[j].split("?")[0]} ? 

                    ### Reasoning: {reasoning[j]}
                    
                    ### Response: 
                ''')
                questions_n.append(questions[j])
                answers_n.append(answers[j])
                reasoning_n.append(reasoning[j])
                cot_prev.append(outputs_0[j])
                tags_n.append(tags[j])
        # breakpoint()
        # get sub questions generated
        # print("INPUTS LLAMA", outputs[0])
        if(len(outputs)>0):
            response = respond(model, outputs)
            # breakpoint()
            # print("RESPONSE LLAMA", response)
            
            s2 = time.time()
            # print("GOT LLAMA REPONSE", s2-s1)
            # sys.exit()
            # # breakpoint()
            
            query_tensor = tokenizer(outputs, return_tensors="pt", padding = True, max_length = max_len).input_ids
            response_tensor = tokenizer(response, return_tensors="pt", padding = True, max_length = max_len).input_ids

            # define a reward for response
            dones, outputs, rewards, cot = env.step(False, response, questions_n, answers_n, dones_0, reasoning_n, cot_prev)

            s3 = time.time()
            
            for i in range(len(dones)):
                if(dones[i]):
                    if(T[i]=="MATH"):
                        solved_1[0]+=1
                    if(T[i]=="AQuA"):    
                        solved_1[1]+=1
                    if(T[i]=="SQA"):
                        solved_1[2]+=1
                    print("################IMPROEMENT#################\n", questions_n[i], "\n", cot_prev[i], "\n", response[i], "\n", outputs[i], "\n", rewards[i])
                # else:
                    # print("################NO IMPROEMENT#################\n", questions_n[i], "\n", cot_prev[i], "\n", response[i], "\n", outputs[i], "\n", rewards[i])

            
            # wandb.log({'Depreciation': acc1, 'Improvement': acc2, 'Both correct': acc3})
            wandb.log({'Base': solved_0, 'Improvement': solved_1})

            
            print(solved_0, solved_1)
            # sys.exit()
            # print("DONE SECOND PASS", s3-s2)

            # print("OUTPUTS", outputs[0])
            # print("REWARDS", rewards[0])
            reward_epoch += sum(rewards)
            # # breakpoint()
            # print("DONES", dones, "\n OUTPUTS", outputs, "\n REWARDS", rewards)
            # break
            bs_pseudo = 0
            len_initial = len(response)
            tensor_query = []
            tensor_response = []
            tensor_rewards = []
            while bs_pseudo<batch_size:
                tensor_query.append(torch.tensor(query_tensor[bs_pseudo%len_initial]))
                tensor_response.append(torch.tensor(response_tensor[bs_pseudo%len_initial]))
                tensor_rewards.append(rewards[bs_pseudo%len_initial])
                bs_pseudo+=1
            # breakpoint()
            # # breakpoint()
            # print_gpu_utilization()
            # Input to llama, output of llama, reward given on thatt
            # # breakpoint()
            train_stats = ppo_trainer.step(tensor_query, tensor_response, tensor_rewards)
            # ppo_trainer.log_stats(train_stats, {'query' : [],'response' :[]}, rewards)
            s4 = time.time()
        # breakpoint()
        r.append(reward_epoch)
        # print(r, rewards)
        batch_number += 1

wandb.finish()