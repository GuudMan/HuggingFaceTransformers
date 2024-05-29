# 1、import modules
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# 2、load datasets
datasets = load_dataset("csv", 
                        data_files="../data/ChnSentiCorp_htl_all.csv", 
                        split="train")
# 2.1 dataset filters
datasets = datasets.filter(lambda x: x['review'] is not None)

# 3、split datasets
datasets = datasets.train_test_split(test_size=0.15)
print(datasets)
print(datasets['train'][0])

# 4、create dataloader
# traindataloader = DataLoader

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
tokenizer.save_pretrained("./5_tx_cls_metrics")

def process_func(examples):
    tokenizer_examples = tokenizer(examples['review'], max_length=128, 
                                   truncation=True)
    tokenizer_examples['labels'] = examples['label']
    return tokenizer_examples

tokenized_datasets = datasets.map(process_func, 
                                  batched=True, 
                                  remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
print(tokenized_datasets['train'][0])

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
trainset, validset = tokenized_datasets['train'], tokenized_datasets['test']
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, 
                         collate_fn=DataCollatorWithPadding(tokenizer))
validloader = DataLoader(validset, batch_size=64, shuffle=False, 
                         collate_fn=DataCollatorWithPadding(tokenizer))

print("-------dataloader------")
# print(next(enumerate(trainloader))[1])
"""
{'input_ids': tensor([[ 101, 2791, 7313,  679, 7231, 8024, 6820, 3300,  702, 7345, 1378, 8024,
    5543, 4692, 1168, 1920, 7368, 4638, 3250, 5682, 8024,  679, 6814, 7345,
    1378,  722, 7313, 4638, 7313, 7392, 1922, 6818,  749, 8024,  886,  782,
    1912, 1139, 4638, 3198,  952, 6230, 2533,  679, 1922, 2128, 1059, 8024,
    852, 4384, 1862, 3221,  671, 3837, 4638,  511, 3315, 3341, 5381,  677,
    3221, 6432, 3300, 3322, 1767, 2970, 6843, 4638, 8024,  679, 6814, 3766,
    3300, 3209, 4802, 4638, 2900, 2471, 2582,  720, 6370, 8024, 1168, 3635,
    722, 1400, 6963, 1372, 5543, 5632, 2346, 2802, 6756, 1343, 6983, 2421,
    511,  102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0],
    [ 101, 6821, 3613, 1057,  857, 2208, 3360, 2161, 7667, 6820, 3221, 3683,
    6772, 4007, 2692, 4638,  117, 2161, 7667, 4638,  855, 5390, 3683, 6772,
    1962, 2823,  117,  794, 2145, 6817, 4991, 2802, 6956, 6369, 4923, 6756,
    1372, 6206,  676, 1126, 1146, 7164,  117,  126, 1779, 7178, 2218, 1377,
    809, 1168,  749,  117, 2791, 7313, 3683, 6772, 2397, 1112,  117, 5050,
    3221, 3683, 6772, 2160, 3139, 4638,  117, 3819, 4074, 4638, 4178, 3717,
    738, 2923, 5653, 3302,  117, 3297, 1962, 4638, 1765, 3175, 3221, 2161,
    7667, 1912, 7481, 3300,  671, 3340, 2523, 3200, 4638, 2207, 1391, 6125,
    117, 3241,  677,  127, 4157, 2340, 1381, 2218, 2458,  749,  117, 2523,
    1914, 7937, 6793, 4176,  722, 5102, 4638, 2207, 1391,  117, 5445,  684,
    3300, 6629, 4772,  753,  676, 1282, 3440,  102]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1]]), 'labels': tensor([1, 1, 1, 1])}
"""

# 5、create model and optimizer
import torch
from torch.optim import Adam
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
if torch.cuda.is_available():
    model = model.cuda()
optimizer = Adam(model.parameters(), lr=2e-5)

# 6、train and evaluation
import evaluate
clf_metrics = evaluate.combine(['accuracy', 'f1'])

def evaluate():
    model.eval()
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            clf_metrics.add_batch(predictions=pred.long(), 
                                  references=batch['labels'].long())
    return clf_metrics.compute()

def train(epochs=3, log_step=100):
    global_step = 0
    for ep in range(epochs):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print("ep:", ep, "global_step： ", 
                      global_step, "loss: ", 
                      output.loss.item())
            global_step += 1
        cfl = evaluate()
        print("ep: ", ep, "clf_metrics：", cfl)

# 7、train
train(epochs=10)
model.save_pretrained("./5_tx_cls_metrics")


# 8、evaluate
# 5_0_tx_cls_evaluate_inference.py






