from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# 1. load dataset
dataset = load_dataset("csv", data_files="./data/ChnSentiCorp_htl_all.csv", split="train")

dataset = dataset.filter(lambda x:x["review"] is not None)


# 2、dataset split
datasets = dataset.train_test_split(test_size=0.1)
print(datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['label', 'review'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['label', 'review'],
        num_rows: 777
    })
})
"""

# 3、Create Dataloader
import torch
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

tokenizer.save_pretrained("./4_tx_cls_model")

def process_function(examples):
    tokenized_examples = tokenizer(examples['review'], 
                                   max_length=128, 
                                   truncation=True)
    tokenized_examples['labels'] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, 
                                 batched=True, 
                                 remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 6988
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 777
    })
})
"""

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

trainset, validset = tokenized_datasets['train'], tokenized_datasets['test']
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, 
                         collate_fn=DataCollatorWithPadding(tokenizer))

validloader = DataLoader(validset, batch_size=64, shuffle=False, 
                         collate_fn=DataCollatorWithPadding(tokenizer))

print(next(enumerate(validloader))[1])
"""
{'input_ids': 
tensor([[ 101, 6983, 2421,  ..., 2421, 4638,  102],
        [ 101, 4692, 3025,  ..., 3322, 3221,  102],
        [ 101, 2600, 5445,  ...,    0,    0,    0],
        ...,
        [ 101, 3766, 3300,  ...,    0,    0,    0],
        [ 101,  857, 2162,  ..., 2421, 8024,  102],
        [ 101,  857, 6814,  ...,    0,    0,    0]]), 
'token_type_ids': 
tensor([[0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        ...,
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0]]), 
'attention_mask': 
tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]]), 
'labels': 
tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0])}
"""


# 4、creat model and optimizer
from torch.optim import Adam
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num // len(validset)

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}
            
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print("ep: ", ep, "global_step:", global_step, "loss:", output.loss.item())
            global_step += 1
        acc = evaluate()
        print("ep: ", ep, "acc: ", acc)
    

train()
model.save_pretrained("./4_tx_cls_model")








