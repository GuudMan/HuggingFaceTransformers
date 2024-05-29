# 1、 import packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 2、 load datasets
print('-------# 2、 load datasets-------')
dataset = load_dataset("json", data_files="./train_pair_1w.json", split="train")
print(dataset)
"""
Dataset({
    features: ['sentence1', 'sentence2', 'label'],
    num_rows: 10000
})
"""
print(dataset[0])
"""
{'sentence1': '找一部小时候的动画片', 
'sentence2': '求一部小时候的动画片。谢了', 
'label': '1'}
"""


# 3、datasets split
datasets = dataset.train_test_split(test_size=0.2)
print(datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label'],
        num_rows: 8000
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label'],
        num_rows: 2000
    })
})
"""

# 4、datasets preprocess
import torch
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
def process_func(examples):
    tokenized_examples = tokenizer(examples['sentence1'], 
                                   examples['sentence2'], 
                                   max_length=128, 
                                   truncation=True)
    tokenized_examples['labels'] = [float(label) for label in examples['label']]
    return tokenized_examples
tokenized_datasets = datasets.map(process_func, batched=True, 
                                  remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 8000
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 2000
    })
})
"""

# 5、create model
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-base", 
                                                           num_labels=1)


# 6、create evaluate function
import evaluate
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p > 0.5) for p in predictions]
    label = (int(l)for l in labels)
    acc = acc_metric.compute(predictions=predictions, 
                             references=labels)
    f1 = f1_metric.compute(predictions=predictions, 
                           references=labels)
    acc.update(f1)
    return acc


# 7、configure TrainingArguments
train_args = TrainingArguments(output_dir="./cross_model_tx_similarity",  # 输出文件夹
                               per_device_train_batch_size=64,     # 训练batch_size
                               per_device_eval_batch_size=128,     # 验证batch_size
                               logging_steps=10,                   # log打印频率
                               evaluation_strategy="epoch",        # 评估策略
                               save_strategy="epoch",              # 保存策略
                               save_total_limit=3,                 # 最大保存数
                               learning_rate=2e-5,                 # 学习率
                               weight_decay=0.01,                  # weight_decay
                               metric_for_best_model="f1",         # 设定评估指标
                               load_best_model_at_end=True         # 训练完成后加载最优模型
                               )

# 8、configure Trainer
from transformers import DataCollatorWithPadding
trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets['train'], 
                  eval_dataset=tokenized_datasets['test'], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer), 
                  compute_metrics=eval_metric
                  )

# 9、model train
trainer.train()

# 10、model prediction
