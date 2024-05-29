# 1、 import packages
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


# 2、 load datasets
print('--------# 2、 load datasets------')
ds = Dataset.load_from_disk("./nlpcc_2017")


# 3、datasets split
print('--------# 3、datasets split------')
ds = ds.train_test_split(100, seed=42)
print(ds['train'][0])

# 4、datasets preprocess
print('--------# 4、datasets preprocess------')
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-large-chinese", trust_remote_code=True)
# def process_func(examples):
#     contents = ["摘要生成: \n" + e + tokenizer.mask_token for e in examples['content']]
#     inputs = tokenizer(contents, max_length=384, truncation=True, 
#                        padding="max_length", return_tensors="pt")
#     inputs = tokenizer.build_inputs_for_generation(inputs, 
#                                                    targerts=examples['title'], 
#                                                    padding=True, 
#                                                    max_gen_length=64)
#     return inputs
# tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds['train'].column_names)
# print(tokenizer.decode(tokenized_ds['train'][0]['input_ids']))
# print(tokenizer.decode(tokenized_ds['train'][0]['labels']))
# print(tokenizer.decode(tokenized_ds['train'][0]['position_ids']))

# 5、create model
print('--------# 5、create model------')


# 6、create evaluate function
print('--------# 6、create evaluate function------')


# 7、configure TrainingArguments
print('--------# 7、configure TrainingArguments------')


# 8、configure Trainer
print('--------# 8、configure Trainer------')


# 9、model train
print('--------# 9、model train------')


# 10、model prediction
print('--------# 10、model prediction------')