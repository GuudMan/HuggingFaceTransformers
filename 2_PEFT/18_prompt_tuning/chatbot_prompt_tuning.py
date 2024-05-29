# 1、 import packages
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 2、 load datasets
print('--------# 2、 load datasets------')
ds = Dataset.load_from_disk("../../1_trans_task/16_generate_chatbot/alpaca_data_zh")
print(ds[:3])


# 3、datasets split
print('--------# 3、datasets split------')



# 4、datasets preprocess
print('--------# 4、datasets preprocess------')
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
def process_func(examples):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], 
                                       examples["input"]]).strip() + "\n\nAssistant: ")
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + tokenizer.eos_token
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }
tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(tokenized_ds[1]["input_ids"])
tokenizer.decode(list(filter(lambda x: x!= -100, tokenized_ds[1]['labels'])))


# 5、create model
print('--------# 5、create model------')
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)



# 6、Prompt tuning
print('--------# 6、create evaluate function------')
# 6.1 configure file


# 7、configure TrainingArguments
print('--------# 7、configure TrainingArguments------')


# 8、configure Trainer
print('--------# 8、configure Trainer------')


# 9、model train
print('--------# 9、model train------')


# 10、model prediction
print('--------# 10、model prediction------')