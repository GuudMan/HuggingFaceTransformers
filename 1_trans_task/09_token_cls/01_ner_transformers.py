# 1、import packages
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments,DataCollatorForTokenClassification
from datasets import DatasetDict

# 2、load datasets
# if network is available, use load_dataset directly， or use load_from_disk
ner_datasets = DatasetDict.load_from_disk("/home/oem/pn/01_transformers/1_trans_task/09_token_cls/ner_data")
print(ner_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 4637
    })
})
"""
print(ner_datasets['train'][0])
"""
{'id': '0', 
'tokens': ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'], 
'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]}
"""
print(ner_datasets['train'].features)
"""
{'id': Value(dtype='string', id=None), 
'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 
'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None)}
"""

label_list = ner_datasets['train'].features['ner_tags'].feature.names
print(label_list)
"""
['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
"""

# 3、datasets preprocess
print("------------# 3、datasets preprocess--------------")
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
tokenizer(ner_datasets['train'][0]['tokens'], is_split_into_words=True)
tokenizer.save_pretrained("./ner_cls_token")
print(tokenizer)
"""
BertTokenizerFast(name_or_path='hfl/chinese-macbert-base', 
vocab_size=21128, model_max_length=1000000000000000019884624838656, 
is_fast=True, padding_side='right', truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]',
'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  
added_tokens_decoder={
        0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
"""
res = tokenizer("interesting word")
print(res)
"""
{'input_ids': [101, 10673, 12865, 12921, 8181, 8681, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
"""
print(res.word_ids)
"""
<bound method BatchEncoding.word_ids of 
{'input_ids': [101, 10673, 12865, 12921, 8181, 8681, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1]}>
"""

# 3.2 label map
print('------------# 3.2 label map------------')
def process_function(examples):
    tokenized_examples = tokenizer(examples['tokens'], max_length=128, 
                                   truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        # 得到每个句子中每个字的索引， 首位两个是None
        word_ids = tokenized_examples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                # 根据索引， 在ner_tags中找到该字对应的标签
                label_ids.append(label[word_id])
        labels.append(label_ids)
    # tokenized_examples中添加标签
    tokenized_examples['labels'] = labels
    return tokenized_examples


tokenized_datasets = ner_datasets.map(process_function, batched=True)
print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 4637
    })
})

"""
print(tokenized_datasets["train"][0])
"""
{'id': '0', 
'tokens': ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'],
'ner_tags': [0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0], 
'input_ids': [101, 3862, 7157, 3683, 6612, 1765, 4157, 1762, 1336, 7305, 680, 7032, 7305, 722, 7313, 4638, 3862, 1818, 511, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'labels': [-100, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0, -100]}
"""

# 4、create model
print('------------# 4、create model------------')
model = AutoModelForTokenClassification.from_pretrained("hfl/chinese-macbert-base", 
                                                        num_labels=len(label_list))
print(model.config.num_labels)

import evaluate
# 5、create evaluate functions
# acc_metric = evaluate.load("accuracy")
# f1_metric = evaluate.load("f1")
# def eval_metric(eval_predict):
#     predictions, labels = eval_predict
#     predictions = predictions.argmax(axis=-1)
#     acc = acc_metric.compute(predictions=predictions, references=labels)
#     f1 = f1_metric.compute(predictions=predictions, references=labels)
#     acc.update(f1)
#     return acc
seqeval = evaluate.load("seqeval_metric.py")
import numpy as np

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels) 
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels) 
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

    return {
        "f1": result["overall_f1"]
    }
    

# 6、TrainingArguments
args = TrainingArguments(
    output_dir="models_for_ner", 
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=128, 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    save_total_limit=3, 
    metric_for_best_model="f1",
    load_best_model_at_end=True, 
    logging_steps=50, 
    num_train_epochs=20
)

# 7、create Trainer
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'], 
    compute_metrics=eval_metric, 
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)


# 8、model train
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets['test'])


# 9、model infer
print("------------# 9、model infer--------------")
from transformers import pipeline
model.config.id2label = {idx:label for idx, label in enumerate(label_list)}
print(model.config)

ner_pipe = pipeline("token-classification", 
                    model=model, 
                    tokenizer=tokenizer, 
                    device=0, 
                    aggregation_strategy="simple")
res = ner_pipe("小明在北京上学")
print(res)

# take out the actual results
ner_result = {}
x = "小明在北京上学"
for r in res:
    if r['entity_group'] not in ner_result:
        ner_result[r['entity_group']] = []
    ner_result[r['entity_group']].append(x[r['start']:r['end']])

print(ner_result)
"""
[{'entity_group': 'PER', 'score': 0.6836417, 'word': '小 明', 'start': 0, 'end': 2},
 {'entity_group': 'LOC', 'score': 0.9998947, 'word': '北 京', 'start': 3, 'end': 5}]
{'PER': ['小明'], 'LOC': ['北京']}
"""