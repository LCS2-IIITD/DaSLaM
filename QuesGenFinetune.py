from datasets import load_dataset
from transformers import LLaMATokenizer
import sys
import os
import re
os.environ['WANDB_DISABLED'] = 'true'
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LLaMAForCausalLM, LLaMATokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model

# Setting for A100 - For 3090 
MICRO_BATCH_SIZE = 4  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 8  # paper uses 3
LEARNING_RATE = 2e-5  # from the original paper
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
           
tokenizer = LLaMATokenizer.from_pretrained("13B_HF/tokenizer.model", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = LLaMAForCausalLM.from_pretrained(
    "13B_HF/",
    load_in_8bit=True,
    device_map={"":0},
)

model = prepare_model_for_int8_training(model)
# sys.exit()

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


def generate_prompt(data_point):    
    if(data_point['question']):
        return f"""Below is an instruction that describes a task, paired with an input and a reasoning that provides further context. Write a response that appropriately completes the request.

    ### Instruction: Break the input question into multiple subquestions based on the reasoning provided. 

    ### Input:
    {data_point["question"]}

    ### Reasoning:
    {data_point["Reasoning"]}

    ### Response:
    {data_point["sub-questions"]}"""


data = load_dataset("json", data_files="merged.json")

data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)


# breakpoint()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        # logging_steps=1,
        output_dir="lora-alpaca-13B-context-qa",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("lora-alpaca-13B-context-qa")

