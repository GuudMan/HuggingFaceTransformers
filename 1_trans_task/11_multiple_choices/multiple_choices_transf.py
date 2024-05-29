# 1、import package
from typing import Any
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer

# 2、load datasets
print('----------# 2、load datasets----------')
c3 = DatasetDict.load_from_disk("./c3")
print(c3)
"""
DatasetDict({
    test: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer'],
        num_rows: 1625
    })
    train: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer'],
        num_rows: 11869
    })
    validation: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer'],
        num_rows: 3816
    })
})
"""
print(c3['train'][:10])
"""
{'id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'context': [['男：你今天晚上有时间吗?我们一起去看电影吧?', '女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……'], ['男：足球比赛是明天上午八点开始吧?', '女：因为天气不好，比赛改到后天下午三点了。'], ['女：今天下午的讨论会开得怎么样?', '男：我觉得发言的人太少了。'], ['男：我记得你以前很爱吃巧克力，最近怎么不吃了，是在减肥吗?', '女：是啊，我希望自己能瘦一点儿。'], ['女：过几天刘明就要从英国回来了。我还真有点儿想他了，记得那年他是刚过完中秋节走的。', '男：可不是嘛!自从我去日本留学，就再也没见过他，算一算都五年了。', '女：从2000年我们在学校第一次见面到现在已经快十年了。我还真想看看刘明变成什么样了!', '男：你还别说，刘明肯定跟英国绅士一样，也许还能带回来一个英国女朋友呢。'], ['男：好久不见了，最近忙什么呢?', '女：最近我们单位要搞一个现代艺术展览，正忙着准备呢。', '男：你们不是出版公司吗?为什么搞艺术展览?', '女：对啊，这次展览是我们出版的一套艺术丛书的重要宣传活动。'], ['男：会议结束后，你记得把空调和灯都关了。', '女：好的，我知道了，明天见。'], ['男：你出国读书的事定了吗?', '女：思前想后，还拿不定主意呢。'], ['男：这件衣服我要了，在哪儿交钱?', '女：前边右拐就有一个收银台，可以交现金，也可以刷卡。'], ['男：小李啊，你是我见过的最爱干净的学生。', '女：谢谢教授夸奖。不过，您是怎么看出来的?', '男：不管我叫你做什么，你总是推得干干净净。', '女：教授，我……']], 'question': ['女的最喜欢哪种电影?', '根据对话，可以知道什么?', '关于这次讨论会，我们可以知道什么?', '女的为什么不吃巧克力了?', '现在大概是哪一年?', '女的的公司为什么要做现代艺术展览?', '他们最可能是什么关系?', '女的是什么意思?', '他们最可能在什么地方?', '教授认为小李怎么样?'], 'choice': [['恐怖片', '爱情片', '喜剧片', '科幻片'], ['今天天气不好', '比赛时间变了', '校长忘了时间'], ['会是昨天开的', '男的没有参加', '讨论得不热烈', '参加的人很少'], ['刷牙了', '要减肥', '口渴了', '吃饱了'], ['2005年', '2010年', '2008年', '2009年'], ['传播文化', '宣传新书', '推广现代艺术', '体现企业文化'], ['同事', '司机和客人', '医生和病人'], ['不想出国', '出国太难', '还在犹豫', '不想决定'], ['医院', '迪厅', '商场', '饭馆'], ['卫生习惯非常好', '做事的能力不够', '找借口拒绝做事', '记不住该做的事']], 'answer': ['喜剧片', '比赛时间变了', '讨论得不热烈', '要减肥', '2010年', '宣传新书', '同事', '还在犹豫', '商场', '找借口拒绝做事']}
"""
print(c3.pop("test"))
print(c3)
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer'],
        num_rows: 11869
    })
    validation: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer'],
        num_rows: 3816
    })
})
"""


# 3、datasets preprocess
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
def process_func(examples):
    # examples: dict, keys: ["context", "question", "choice", "answer"]
    context = []
    question_choice = []
    labels = []
    for idx in range(len(examples['context'])):
        ctx = "\n".join(examples['context'][idx])
        question = examples['question'][idx]
        choices = examples['choice'][idx]
        for choice in choices:
            context.append(ctx)
            question_choice.append(question + " " + choice)
        if len(choices) < 4:
            for _ in range(4 - len(choices)):
                context.append(ctx)
                question_choice.append(question + " " + "不知道")
        labels.append(choices.index(examples['answer'][idx]))
    tokenized_examples = tokenizer(context, 
                                   question_choice, 
                                   truncation="only_first", 
                                   max_length=256, 
                                   padding="max_length")
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] 
                          for k, v in tokenized_examples.items()}
    tokenized_examples['labels'] = labels
    return tokenized_examples

print('--------------2.1 c3--------------')
res = c3['train'].select(range(10)).map(process_func, batched=True)
print(res)
"""
Dataset({
    features: ['id', 'context', 'question', 'choice', 'answer', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
    num_rows: 10
})
"""


print('--------------2.2 tokenized_c3--------------')
import numpy as np
print(np.array(res['input_ids']).shape)
tokenized_c3 = c3.map(process_func, batched=True)
print(tokenized_c3)
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 11869
    })
    validation: Dataset({
        features: ['id', 'context', 'question', 'choice', 'answer', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 3816
    })
})
"""
print('-------------2.3 c3------tokenized_c3----------')
print(c3['train'][0])
"""
{'id': 0, 
'context': ['男：你今天晚上有时间吗?我们一起去看电影吧?', '女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……'], 
'question': '女的最喜欢哪种电影?', 
'choice': ['恐怖片', '爱情片', '喜剧片', '科幻片'],
'answer': '喜剧片'}

"""
# print(tokenized_c3['train'][0])
"""
-------------
{'id': 0, 
'context': ['男：你今天晚上有时间吗?我们一起去看电影吧?', '女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……'], 
'question': '女的最喜欢哪种电影?', 
'choice': ['恐怖片', '爱情片', '喜剧片', '科幻片'], 
'answer': '喜剧片', 
'input_ids': [[101, 4511, 8038, 872, 791, 1921, 3241, 677, 3300, 3198, 7313, 
1408, 136, 2769, 812, 671, 6629, 1343, 4692, 4510, 2512, 1416, 136, 1957,
 8038, 872, 1599, 3614, 2607, 2587, 4275, 1469, 4263, 2658, 4275, 8024, 852, 
 3221, 2769, 1599, 3614, 1599, 1196, 4275, 8024, 4906, 2404, 4275, 671, 5663, 
 511, 2792, 809, 100, 100, 102, 1957, 4638, 3297, 1599, 3614, 1525, 4905, 4510, 
 2512, 136, 2607, 2587, 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[101, 4511, 8038, 872, 791, 1921, 3241, 677, 3300, 3198, 7313, 1408, 136, 2769, 812, 
671, 6629, 1343, 4692, 4510, 2512, 1416, 136, 1957, 8038, 872, 1599, 3614, 2607, 
2587, 4275, 1469, 4263, 2658, 4275, 8024, 852, 3221, 2769, 1599, 3614, 1599, 1196, 
4275, 8024, 4906, 2404, 4275, 671, 5663, 511, 2792, 809, 100, 100, 102, 1957, 4638, 
3297, 1599, 3614, 1525, 4905, 4510, 2512, 136, 4263, 2658, 4275, 102, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 [101, 4511, 8038, 872, 791, 1921, 3241, 677, 3300, 3198, 7313, 1408, 136, 2769, 
 812, 671, 6629, 1343, 4692, 4510, 2512, 1416, 136, 1957, 8038, 872, 1599, 3614, 
 2607, 2587, 4275, 1469, 4263, 2658, 4275, 8024, 852, 3221, 2769, 1599, 3614, 
 1599, 1196, 4275, 8024, 4906, 2404, 4275, 671, 5663, 511, 2792, 809, 100, 100, 
 102, 1957, 4638, 3297, 1599, 3614, 1525, 4905, 4510, 2512, 136, 1599, 1196, 
 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0], 
 [101, 4511, 8038, 872, 791, 1921, 3241, 677, 3300, 3198, 7313, 1408, 136, 2769, 
 812, 671, 6629, 1343, 4692, 4510, 2512, 1416, 136, 1957, 8038, 872, 1599, 3614, 
 2607, 2587, 4275, 1469, 4263, 2658, 4275, 8024, 852, 3221, 2769, 1599, 3614, 
 1599, 1196, 4275, 8024, 4906, 2404, 4275, 671, 5663, 511, 2792, 809, 100, 100, 
 102, 1957, 4638, 3297, 1599, 3614, 1525, 4905, 4510, 2512, 136, 4906, 2404, 
 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'labels': 2}
"""

# 4、create model
model = AutoModelForMultipleChoice.from_pretrained("hfl/chinese-macbert-base")


# 5、create evaluate functions
import numpy as np
import evaluate
accuracy = evaluate.load("accuracy")
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 6、configure TrainingArguments
args = TrainingArguments(
    output_dir="./multiple_choice", 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64, 
    num_train_epochs=3, 
    logging_steps=50, 
    evaluation_strategy="epoch",
    save_strategy="epoch", 
    load_best_model_at_end=True, 
    fp16=True
)

# 7、configure Trainer
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_c3['train'], 
    eval_dataset=tokenized_c3['validation'],
    compute_metrics=compute_metrics
)


# 8、model train
trainer.train()

# 9、model evaluate
import torch
print('----------# 9、model evaluate-----------')
class MultipleChoicePipeline:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def preprocess(self, context, question, choices):
        cs, qcs = [], []
        for choice in choices:
            cs.append(context)
            qcs.append(question + " " + choice)
        return tokenizer(cs, qcs, truncation="only_first",
                          max_length=256, 
                         return_tensors="pt")
    
    def predict(self, inputs):
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits

    def postprocess(self, logits, choices):
        prediction = torch.argmax(logits, dim=-1).cpu().item()
        return choices[prediction]

    def __call__(self, context, question, choices):
        inputs = self.preprocess(context, question, choices)
        logits = self.predict(inputs)
        result = self.postprocess(logits, choices)
        return result

pipe = MultipleChoicePipeline(model, tokenizer)
res = pipe(context="小明在北京", question="小明在哪个城市", choices=['北京', '上海', '河北', '海南'])
print(res)