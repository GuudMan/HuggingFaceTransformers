# 1、import packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
# from torch.utils.data import 
from torch.optim import Adam

# 2、load datasets
datasets = load_dataset("csv", 
                        data_files="../data/ChnSentiCorp_htl_all.csv", 
                        split="train")
datasets = datasets.filter(lambda x: x["review"] is not None)
# 3、split datasets
datasets = datasets.train_test_split(test_size=0.15)
# print(datasets)

# 4、dataset preprocess
import torch
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
# tokenizer.save_pretrained("./6_trainer_tx_cls")

def process_func(examples):
    tokenized_examples = tokenizer(examples['review'], 
                                   max_length=128, 
                                   truncation=True)
    tokenized_examples['labels'] = examples['label']
    return tokenized_examples

tokenized_datasets = datasets.map(process_func, batched=True, 
                                  remove_columns=datasets['train'].column_names)

# 5、create model
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model.cuda()

# 6、create evaluation functions
import evaluate
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
def eval_metric(eval_prediction):
    predictions, labels = eval_prediction
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, 
                             references=labels)
    f1 = f1_metric.compute(predictions=predictions, 
                           references=labels)
    acc.update(f1)
    return acc

# 7、create TrainingArgments
train_args = TrainingArguments(output_dir="6_0_trainer_tx_cls",  # 输出文件夹
                               per_device_train_batch_size=64,   # 训练batch_size
                               per_gpu_eval_batch_size=128,      # 验证batch_size
                               logging_steps=10,                 # log 打印频率
                               evaluation_strategy="epoch",      # 评估策略
                               save_strategy="epoch",            # 保存策略
                               save_total_limit=3,               # 最大保存数
                               learning_rate=2e-5,               # 学习率
                               weight_decay=0.01,                # weight_decay
                               metric_for_best_model="f1",       # 设定评估指标
                               load_best_model_at_end=True       # 训练完成后加载最优模型
                               )


# 8、create Trainer
from transformers import DataCollatorWithPadding
trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets['train'], 
                  eval_dataset=tokenized_datasets['test'], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer), 
                  compute_metrics=eval_metric, 
                  )

# 9、train
trainer.train()

# 10、evaluate
trainer.evaluate(tokenized_datasets['test'])

# 11、train and evaluate
model.save_pretrained("./6_trainer_tx_cls")





