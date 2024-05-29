# 1、import packages
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import load_dataset, DatasetDict
# 2、load_datasets
datasets = DatasetDict.load_from_disk("./mrc_data")
print(datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['id', 'context', 'question', 'answers'],
        num_rows: 10142
    })
    validation: Dataset({
        features: ['id', 'context', 'question', 'answers'],
        num_rows: 3219
    })
    test: Dataset({
        features: ['id', 'context', 'question', 'answers'],
        num_rows: 1002
    })
})
"""
print(datasets['train'][0])
"""
{'id': 'TRAIN_186_QUERY_0', 
'context': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。
1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；
1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。
范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，
被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。
范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。
1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，
法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，
但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、
自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天
主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越
南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷
颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会
团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗
座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为
天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢
升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4
月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河
内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴
光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世
，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。', 
'question': '范廷颂是什么时候被任为主教的？', 
'answers': {'text': ['1963年'], 
'answer_start': [30]}}
"""

# 3、dataset preprocess
print("-------# 3、dataset preprocess---------")
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
sample_dataset = datasets['train'].select(range(10))
print(sample_dataset)
"""
Dataset({
    features: ['id', 'context', 'question', 'answers'],
    num_rows: 10
})
"""
tokenized_examples = tokenizer(text=sample_dataset['question'], 
                               text_pair=sample_dataset['context'], 
                               return_offsets_mapping=True, 
                               max_length=512, truncation="only_second", 
                               padding="max_length")
# print(tokenized_examples)
print(tokenized_examples.keys())
"""
dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping'])
"""
# print(tokenized_examples['offset_mapping'][0], 
#       len(tokenized_examples['offset_mapping'][0]))

# -------sequence_ids--其实就是token_type_ids---
print("# -------sequence_ids-----")
print(tokenized_examples.sequence_ids(0))
"""
[None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None]
"""

# 3.2 offser_mapping
print('---------# 3.2 offser_mapping-----------')
offset_mapping = tokenized_examples.pop('offset_mapping')
for idx, offset in enumerate(offset_mapping):
    answer = sample_dataset[idx]['answers']
    start_char = answer['answer_start'][0]
    end_char = start_char + len(answer['text'][0])
    # 定位答案在token中的起始位置和结束位置
    # 一种策略，拿到context的起始和结束， 然后从左右两侧向答案逼近

    context_start = tokenized_examples.sequence_ids(idx).index(1)
    context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1

    # 判断答案是否在context中
    if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
        start_token_pos = 0
        end_token_pos = 0
    else:
        token_id = context_start
        while token_id <= context_end and offset[token_id][0] < start_char:
            token_id += 1
        start_token_pos = token_id
        token_id = context_end
        while token_id >= context_start and offset[token_id][1] > end_char:
            token_id -= 1
        end_token_pos = token_id
    print(answer, start_char, end_char, context_start, 
          context_end, start_token_pos, end_token_pos)
    print("token answer decode: ", 
          tokenizer.decode(tokenized_examples["input_ids"][idx][start_token_pos: end_token_pos + 1]))

def process_func(examples):
    tokenized_examples = tokenizer(text=examples['question'], 
                                   text_pair=examples['context'], 
                                   return_offsets_mapping=True, 
                                   max_length=384, 
                                   truncation="only_second", 
                                   padding="max_length")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = []
    end_positions = []
    for idx, offset in enumerate(offset_mapping):
        answer = examples["answers"][idx]
        """
        'answers': {'text': ['1963年'], 'answer_start': [30]}}
        """
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])

        context_start = tokenized_examples.sequence_ids(idx).index(1)
        # index方法中， 第一个参数为需要查找的元素， 第二个参数为开始查找的位置
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
        # 判断答案是否在context中
        if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        print(answer, start_char, end_char, context_start, 
              start_token_pos, end_token_pos)
        print("token answer decode:", 
              tokenizer.decode(tokenized_examples['input_ids'][idx][start_token_pos:end_token_pos]))
        

def process_func(examples):
    tokenized_examples = tokenizer(text=examples['question'], 
                                   text_pair=examples['context'], 
                                   return_offsets_mapping=True, 
                                   max_length=384, 
                                   truncation="only_second", 
                                   padding="max_length")    
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = []
    end_positions = []
    for idx, offset in enumerate(offset_mapping):
        answer = examples['answers'][idx]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        # 定位答案在token中的起始位置和结束位置
        # 一种策略， 拿到context的起始和结束， 然后从左右两侧向答案逼近
        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1
        if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
    
    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    return tokenized_examples
tokenized_datasets = datasets.map(process_func, batched=True, 
                                remove_columns=datasets['train'].column_names)
# 3.3 tokenized_datasets
print("--------# 3.3 tokenized_datasets----------")
print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
        num_rows: 10142
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
        num_rows: 3219
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
        num_rows: 1002
    })
})
"""
# print(tokenized_datasets['train'][1])

# 4、load model
model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-macbert-base")


# 5、train arguments
args = TrainingArguments(
    output_dir="model_for_qa", 
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=64, 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    logging_steps=50, 
    num_train_epochs=3
)
# 6、configure Trainer
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['validation'],
    data_collator=DefaultDataCollator()
)


# 7、model train
trainer.train()