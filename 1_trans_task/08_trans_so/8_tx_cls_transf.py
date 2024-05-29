# 1、import packages
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

# 2、load datasets
datasets = load_dataset("csv",
                        data_files="../../data/ChnSentiCorp_htl_all.csv",
                        split="train")
datasets = datasets.filter(lambda x: x['review'] is not None)
# datasets.shuffle()

# 3、datasets split
datasets = datasets.train_test_split(test_size=0.15)

# 4、datasets process
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-large")
tokenizer.save_pretrained("./tx_cls_transf_token")


def process_func(examples):
    tokenizer_examples = tokenizer(examples['review'],
                                   max_length=128,
                                   truncation=True)
    tokenizer_examples['labels'] = examples['label']
    return tokenizer_examples


tokenized_datasets = datasets.map(process_func,
                                  batched=True,
                                  remove_columns=datasets['train'].column_names)

# 5、create model
import torch

model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-macbert-large")
if torch.cuda.is_available():
    model.cuda()

# 6、create evaluation function
import evaluate

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


# 7、create TrainingArguments
train_args = TrainingArguments(output_dir="./tx_cls_transf",  # 输出文件夹
                               per_device_train_batch_size=12,  # 训练时的batch_size
                               gradient_accumulation_steps=32,  # 梯度累加
                               gradient_checkpointing=True,  # 梯度检查点
                               optim="adafactor",  # adafactor优化器
                               per_device_eval_batch_size=12,  # 验证时batch_size
                               num_train_epochs=10,  # 训练轮数
                               logging_steps=10,  # log打印频率
                               evaluation_strategy="epoch",  # 评估策略
                               save_strategy="epoch",  # 保存策略
                               save_total_limit=3,  # 最大保存数
                               learning_rate=2e-5,  # 学习率
                               weight_decay=0.01,  # weight_decay
                               metric_for_best_model="f1",  # 设定评估指标
                               load_best_model_at_end=True  # 训练完成后加载最优模型
                               )

# 8、create Trainer
from transformers import DataCollatorWithPadding

# freeze parameters
for name, param in model.bert.named_parameters():
    param.requires_grad = False
trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['test'],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

# 9、train()
trainer.train()
trainer.evaluate(tokenized_datasets['test'])
trainer.predict(tokenized_datasets['test'])

