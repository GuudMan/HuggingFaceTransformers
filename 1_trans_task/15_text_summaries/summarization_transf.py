# 1、import package
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


# 2、 load dataset
ds = Dataset.load_from_disk("./nlpcc_2017/")
ds = ds.train_test_split(100, seed=42)
print(ds['train'][0])


# 3、data process
tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
print(tokenizer)

def process_func(examples):
    contents = ["摘要生成: \n" + e for e in examples['content']]
    inputs = tokenizer(contents, max_length=384, truncation=True)
    labels = tokenizer(text_target=examples['title'], max_length=64, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_ds = ds.map(process_func, batched=True)
tokenized_ds

print(tokenizer.decode(tokenized_ds['train'][0]['input_ids']))
print(tokenizer.decode(tokenized_ds['train'][0]['labels']))


# 4、 create model
model = AutoModelForSeq2SeqLM.from_pretrained("Langboat/mengzi-t5-base")
model

# 5、evaluation function
import numpy as np
from rouge_chinese import Rouge

rouge = Rouge()

def compute_metric(evalPred):
    predictions, labels = evalPred
    decode_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decode_preds = [" ".join(p) for p in decode_preds]
    decode_labels = [" ".join(l) for l in decode_labels]
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
    return {
        "rouge-1": scores["rouge-1"]['f'], 
        "rouge-2": scores["rouge-2"]['f'], 
        "rouge-l": scores["rouge-l"]['f'], 
    }


#6、 configure training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./summary",
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=8, 
    gradient_accumulation_steps=8, 
    logging_steps=8, 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    metric_for_best_model="rouge-l",
    predict_with_generate=True
)

# 7、create trainer
trainer = Seq2SeqTrainer(
    args=args, 
    model=model, 
    train_dataset=tokenized_ds["train"], 
    eval_dataset=tokenized_ds["test"], 
    compute_metrics=compute_metric, 
    tokenizer=tokenizer, 
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)


# 8、model train
trainer.train()




