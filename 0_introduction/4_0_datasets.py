from datasets import *

# 1、 load dataset online
datasets = load_dataset("madao33/new-title-chinese")
print(datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['title', 'content'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content'],
        num_rows: 1679
    })
})
"""

# 1.1 Loading one of the tasks in the dataset collection 
boolq_dataset = load_dataset("super_glue", "boolq")
print(boolq_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['question', 'passage', 'idx', 'label'],
        num_rows: 9427
    })
    validation: Dataset({
        features: ['question', 'passage', 'idx', 'label'],
        num_rows: 3270
    })
    test: Dataset({
        features: ['question', 'passage', 'idx', 'label'],
        num_rows: 3245
    })
})
"""

# 1.2 load data by split
dataset = load_dataset("madao33/new-title-chinese", split="train")
print(dataset)
"""
Dataset({
    features: ['title', 'content'],
    num_rows: 5850
})
"""

# 1.3 load data by slice
dataset = load_dataset("madao33/new-title-chinese", split="train[10:100]")
print(dataset)
"""
Dataset({
    features: ['title', 'content'],
    num_rows: 90
})
"""

# 1.4 load data by ratio
dataset = load_dataset("madao33/new-title-chinese", split="train[:50%]")
print(dataset)
"""
Dataset({
    features: ['title', 'content'],
    num_rows: 2925
})
"""

# 1.5 view dataset
dataset = load_dataset("madao33/new-title-chinese")
# print(dataset["train"][0])
"""
{'title': '望海楼美国打“台湾牌”是危险的赌博', 
'content': '近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，
中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定
，粗暴干涉中国内政，中方对此坚决反对并已向美方提出严正交涉。\n事实上，
中国高度关注美国国内打“台湾牌”、挑战一中原则的危险动向。近年来，
作为“亲台”势力大本营的美国国会动作不断，先后通过“与台湾交往法”“亚洲再保证倡议法”
等一系列“挺台”法案，“2019财年国防授权法案”也多处触及台湾问题。
今年3月，美参院亲台议员再抛“台湾保证法”草案。众院议员继而在4月提出众院版的草
案并在近期通过。上述法案的核心目标是强化美台关系，并将台作为美“印太战略”的重要伙伴。
同时，“亲台”议员还有意制造事端。今年2月，5名共和党参议员致信众议院议长，
促其邀请台湾地区领导人在国会上发表讲话。这一动议显然有悖于美国与台湾的非官方关系，
其用心是实质性改变美台关系定位。\n上述动向出现并非偶然。在中美建交40周年之际，
两国关系摩擦加剧，所谓“中国威胁论”再次沉渣泛起。美国对华认知出现严重偏差，
对华政策中负面因素上升，保守人士甚至成立了“当前中国威胁委员会”。在此背景下，
美国将台海关系作为战略抓手，通过打“台湾牌”在双边关系中增加筹码。特朗普就任后，
国会对总统外交政策的约束力和塑造力加强。其实国会推动通过涉台法案对行政部门不具约束力，
美政府在2018年并未提升美台官员互访级别，美军舰也没有“访问”台湾港口，保持着某种克制。
但从美总统签署国会通过的法案可以看出，国会对外交产生了影响。
立法也为政府对台政策提供更大空间。\n然而，美国需要认真衡量打“台湾牌”成本。
首先是美国应对危机的代价。美方官员和学者已明确发出警告，美国卷入台湾问题得不偿失。
美国学者曾在媒体发文指出，如果台海爆发危机，美国可能需要“援助”台湾，进而导致新的冷战
乃至与中国大陆的冲突。但如果美国让台湾自己面对，则有损美国的信誉，影响美盟友对同
盟关系的支持。其次是对中美关系的危害。历史证明，中美合则两利、斗则两伤。中美关系是
当今世界最重要的双边关系之一，保持中美关系的稳定发展，不仅符合两国和两国人民的根
本利益，也是国际社会的普遍期待。美国蓄意挑战台湾问题的底线，加剧中美关系的复杂性
和不确定性，损害两国在重要领域合作，损人又害己。\n美国打“台湾牌”是一场危险的赌博。
台湾问题是中国核心利益，中国政府和人民决不会对此坐视不理。中国敦促美方恪守一个中国原
则和中美三个联合公报规定，阻止美国会审议推进有关法案，妥善处理涉台问题。美国悬崖勒马，
才是明智之举。\n（作者系中国国际问题研究院国际战略研究所副所长）'}
"""
print(dataset['train']['title'][:5])
"""
['望海楼美国打“台湾牌”是危险的赌博', 
'大力推进高校治理能力建设', 
'坚持事业为上选贤任能', 
'“大朋友”的话儿记心头', 
'用好可持续发展这把“金钥匙”']
"""
print(dataset['train'].column_names)
"""
['title', 'content']
"""
print(dataset['train'].features)
"""
{'title': Value(dtype='string', id=None), 
'content': Value(dtype='string', id=None)}
"""

# 2、 dataset split
dataset = dataset["train"]
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['title', 'content'],
        num_rows: 5265
    })
    test: Dataset({
        features: ['title', 'content'],
        num_rows: 585
    })
})
"""

# 3、data select and filter
# 3.1 select
dataset = dataset['train'].select([0, 1])
print(dataset)
"""
Dataset({
    features: ['title', 'content'],
    num_rows: 2
})
"""

# 3.2 filter
dataset = load_dataset("madao33/new-title-chinese")
filter_dataset = dataset['train'].filter(lambda example: "中国" in example['title'])
print(filter_dataset['title'][:5])
"""
['聚焦两会，世界探寻中国成功秘诀', 
'望海楼中国经济的信心来自哪里',
'“中国奇迹”助力世界减贫跑出加速度',
'和音瞩目历史交汇点上的中国', 
'中国风采感染世界']
"""

# 4、 data map
def add_prefix(example):
    # add 'Prefix' in front of example
    example['title'] = 'Prefix: ' + example['title']
    return example

# prefix_dataset = datasets.map(add_prefix)
prefix_dataset = dataset.map(lambda x: add_prefix(x))
print(prefix_dataset['train'][:10]['title'])
"""
['Prefix: 望海楼美国打“台湾牌”是危险的赌博', 
'Prefix: 大力推进高校治理能力建设', 
'Prefix: 坚持事业为上选贤任能', 
'Prefix: “大朋友”的话儿记心头', 
...]
"""


# 5、 data preprocessing
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
def preprocess_function(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example['content'], max_length=512, truncation=True)
    labels = tokenizer(example['title'], max_length=32, truncation=True)
    # label就是title编码的结果
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# processed_datasets = datasets.map(preprocess_function)
# print(processed_datasets)
print(datasets['train'][0])
"""

DatasetDict({
    train: Dataset({
        features: ['title', 'content'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content'],
        num_rows: 1679
    })
})

{'title': '望海楼美国打“台湾牌”是危险的赌博', 
'content': '近期，美国国会众院通过法案，重申美国对台湾的承诺。对此，
中国外交部发言人表示，有关法案严重违反一个中国原则和中美三个联合公报规定'}
"""
print("==========processed_datasets==============")
processed_datasets = datasets.map(preprocess_function)
print(processed_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
"""
print(processed_datasets['train'][0])
"""
{'title': '望海楼美国打“台湾牌”是危险的赌博', 
'content': '近期，美国国会众院通过法案，重申美国对台湾的承诺。...）',
'input_ids': [101, 6818, 3309, 8024, 5401, 1744, 1744, 833, 830, 7368, 6858, 6814, 3791, 3428, 8024, 
7028, 4509, 5401, 1744, 2190, 1378, 3968, 4638, 2824, 6437, 511, 2190, 3634, 8024, 704, 1744, 1912,
769, 6956, 1355, 6241, 782, 6134, 4850, 8024, 3300, 1068, 3791, ...], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ], 
'labels': [101, 3307, 3862, 3517, 5401, 1744, 2802, 100, 1378, 3968, 4277, 100, 3221, 1314, 
7372, 4638, 6603, 1300, 102]}
"""

# 5.1 Multi-processing to speed up data preprocessing 
print("=======5.1 Multi-processing to speed up data preprocessing=======")
processed_datasets = datasets.map(preprocess_function, num_proc=4)
print(processed_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
"""

# 5.2 data preprocessing with batched
print("=======5.2 data preprocessing with batched=======")
processed_datasets = datasets.map(preprocess_function, batched=True)
print(processed_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['title', 'content', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
"""

# 5.3 data preprocessing remove columns
print("=======5.3 data preprocessing remove columns=======")
print(datasets['train'].column_names)
"""['title', 'content']"""

processed_datasets = datasets.map(preprocess_function, batched=True, 
                                  remove_columns=datasets['train'].column_names)
print(processed_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 5850
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 1679
    })
})
"""


# 6、 load and save
# 6.1 save
processed_datasets.save_to_disk("./processed_data")

# 6.2 load 
# 4_0_func_load_from_disk.py

# 7、load data from local
# 7.1 load data from local as datasets

print("=======7.1 load data from local as datasets=======")
dataset = load_dataset("csv", data_files="./data/ChnSentiCorp_htl_all.csv", split="train")
print(dataset)
"""
Dataset({
    features: ['label', 'review'],
    num_rows: 7766
})
"""

# 7.2 dataset from csv
print("=======7.1dataset from csv=======")
dataset = Dataset.from_csv("./data/ChnSentiCorp_htl_all.csv")
print(dataset)
"""
Dataset({
    features: ['label', 'review'],
    num_rows: 7766
})
"""

# 7.3 load data with dataframe
print("=======# 7.3 load data with dataframe=======")
import pandas as pd

data = pd.read_csv("./data/ChnSentiCorp_htl_all.csv")
print(data.head(5))
"""
   label                                             review
0      1  距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较...
1      1                       商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!
2      1         早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。房间本身很好。
3      1  宾馆在小街道上，不大好找，但还好北京热心同胞很多~宾馆设施跟介绍的差不多，房间很小，确实挺小...
4      1               CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风
"""

dataset = Dataset.from_pandas(data)
print(dataset)
"""
Dataset({
    features: ['label', 'review'],
    num_rows: 7766
})
"""

# 7.4 dataset with list
data = [{"text": "abc"}, {"text": "def"}]
dataset = Dataset.from_list(data)
print(dataset)
"""
Dataset({
    features: ['text'],
    num_rows: 2
})
"""

# 7.5 load data with self-define function
dataset = load_dataset("json", 
                       data_files="../data/cmrc2018_trial.json", 
                       field="data")
print(dataset)


# 7.6 load data with self-define script
print("=======7.6 load data with self-define script=======")
dataset = load_dataset("./load_script.py", split="train")
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'paragraphs'],
        num_rows: 256
    })
})
"""
print(dataset)
"""
Dataset({
    features: ['id', 'context', 'question', 'answers'],
    num_rows: 1002
})
"""
print(dataset[0])
"""
{'id': 'TRIAL_800_QUERY_0', 
'context': '基于《跑跑卡丁车》与《泡泡堂》上所开发的游戏，由韩国Nexon开发与发行。
中国大陆由盛大游戏运营，这是Nexon时隔6年再次授予盛大网络其游戏运营权。台湾由游戏橘子运营。
玩家以水枪、小枪、锤子或是水炸弹泡封敌人(玩家或NPC)，即为一泡封，将水泡击破为一踢爆。
若水泡未在时间内踢爆，则会从水泡中释放或被队友救援(即为一救援)。每次泡封会减少生命数，生命数耗完即算为踢爆。
重生者在一定时间内为无敌状态，以踢爆数计分较多者获胜，规则因模式而有差异。以2V2、4V4随机配对的方式，
玩家可依胜场数爬牌位(依序为原石、铜牌、银牌、金牌、白金、钻石、大师) ，可选择经典、热血、狙击等模式进行游戏。
若游戏中离，则4分钟内不得进行配对(每次中离+4分钟)。开放时间为暑假或寒假期间内不定期开放，8人经典模式随机配对，
采计分方式，活动时间内分数越多，终了时可依该名次获得奖励。',
'question': '生命数耗完即算为什么？', 
'answers': {'text': ['踢爆'], 'answer_start': [127]}}
"""


# 8 Dataset with DataCollator
from transformers import DataCollatorWithPadding

dataset = load_dataset("csv", 
                       data_files="./data/ChnSentiCorp_htl_all.csv",
                         split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)
print(dataset)
"""
Dataset({
    features: ['label', 'review'],
    num_rows: 7765
})
"""

# 8.1 data map
print("=======# 8.1 data map=======")
def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_dataset = dataset.map(process_function, batched=True, remove_columns=dataset.column_names)
print(tokenized_dataset)
"""
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 7765
})
"""
print(tokenized_dataset[:3])
"""
{'input_ids': 
[[101, 6655, 4895, 2335, 3763, 1062, 6662, 6772, 6818, 117, 852, 3221, 1062, 769, 2900, 4850, 679, 2190, 117, 1963, 3362, 3221, 107, 5918, 7355, 5296, 107, 4638, 6413, 117, 833, 7478, 2382, 7937, 4172, 119, 2456, 6379, 4500, 1166, 4638, 6662, 5296, 119, 2791, 7313, 6772, 711, 5042, 1296, 119, 102], 
[101, 1555, 1218, 1920, 2414, 2791, 8024, 2791, 7313, 2523, 1920, 8024, 2414, 3300, 100, 2160, 8024, 3146, 860, 2697, 6230, 5307, 3845, 2141, 2669, 679, 7231, 106, 102], 
[101, 3193, 7623, 1922, 2345, 8024, 3187, 6389, 1343, 1914, 2208, 782, 8024, 6929, 6804, 738, 679, 1217, 7608, 1501, 4638, 511, 6983, 2421, 2418, 6421, 7028, 6228, 671, 678, 6821, 702, 7309, 7579, 749, 511, 2791, 7313, 3315, 6716, 2523, 1962, 511, 102]], 
'token_type_ids': 
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
'attention_mask': 
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 
'labels': [1, 1, 1]}
"""

collator = DataCollatorWithPadding(tokenizer=tokenizer)
from torch.utils.data import DataLoader
dl = DataLoader(tokenized_dataset, 
                batch_size=4, 
                collate_fn=collator, 
                shuffle=True)

# view the size of input_ids
num = 0
for batch in dl:
    print(batch["input_ids"].size())
    num += 1
    if num > 10:
        break

"""
torch.Size([4, 128])
torch.Size([4, 121])
torch.Size([4, 128])
torch.Size([4, 128])
torch.Size([4, 128])
torch.Size([4, 128])
torch.Size([4, 128])
torch.Size([4, 128])
torch.Size([4, 119])
torch.Size([4, 128])
torch.Size([4, 128])
"""







































