# 1、 import packages
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

# 2、 load datasets
print('--------# 2、 load datasets------')
ds = Dataset.load_from_disk("../../1_trans_task/16_generate_chatbot/alpaca_data_zh")
print(ds)
print(ds[:3])

# 3、datasets split
print('--------# 3、datasets split------')


# 4、datasets preprocess
print('--------# 4、datasets preprocess------')
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh")
tokenizer

def process_func(examples):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples["instruction"], 
                                       examples["input"]]).strip() + "\n\nAssistent: ")
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
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
print(tokenized_ds)
tokenizer.decode(tokenized_ds[1]["input_ids"])
print(tokenizer.decode(list(filter(lambda x: x!= -100, tokenized_ds[1]['labels']))))

# 5、create model
print('--------# 5、create model------')
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-1b4-zh", low_cpu_mem_usage=True)
print(sum(param.numel() for param in model.parameters()))


# 6、BitFit
print('--------# 6、create evaluate function------')
# 选择模型参数里面的所有bias部分
num_param = 0
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()
print(num_param)


# 7、configure TrainingArguments
print('--------# 7、configure TrainingArguments------')
args = TrainingArguments(
    output_dir="./chatbot_1b4", 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8, 
    logging_steps=10, 
    num_train_epochs=1
)


# 8、configure Trainer
print('--------# 8、configure Trainer------')
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_ds, 
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 9、model train
print('--------# 9、model train------')
trainer.train()
model = model.cuda()
ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", 
                                       "").strip() + "\n\nAssistant: ",
                                         return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, 
                                max_length=128, 
                                do_sample=True)[0], 
                                skip_special_tokens=True)


# 10、model prediction
print('--------# 10、model prediction------')
from transformers import pipeline
pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=0)
ipt = "Human:{}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: "
res = pipe(ipt, max_length=256, do_sample=True)
print(res)



