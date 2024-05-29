from transformers import AutoTokenizer

# 1. load and save
# 1.1 load from huggingface
tokenizer = AutoTokenizer.from_pretrained(
    "uer/roberta-base-finetuned-dianping-chinese")
print(tokenizer)
"""
BertTokenizerFast(name_or_path='uer/roberta-base-finetuned-dianping-chinese', 
vocab_size=21128, model_max_length=1000000000000000019884624838656, is_fast=True, 
padding_side='right', truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 
'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
    0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
"""

# 1.2 save tokenizer to local
print(tokenizer.save_pretrained("./roberta_tokenizer"))
"""
('./roberta_tokenizer/tokenizer_config.json', 
'./roberta_tokenizer/special_tokens_map.json', 
'./roberta_tokenizer/vocab.txt', 
'./roberta_tokenizer/added_tokens.json', 
'./roberta_tokenizer/tokenizer.json')
"""

# 1.3 load from local
tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer")
print(tokenizer)
"""
BertTokenizerFast(name_or_path='./roberta_tokenizer', 
vocab_size=21128, model_max_length=1000000000000000019884624838656, 
is_fast=True, padding_side='right', truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 
'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, 
clean_up_tokenization_spaces=True),  added_tokens_decoder={
    0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
    103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
"""

# 2、sentence tokenization
sentence = "中华民族是一个伟大的民族， 中国共产党是一个伟大、光荣、正确的党!"
tokens = tokenizer.tokenize(sentence)
print(tokens)
"""
['中', '华', '民', '族', '是', '一', '个', '伟', '大', '的', '民', '族', '，', '中', '国', '共', '产', '党', '是', '一', '个', '伟', '大', '、', '光', '荣', '、', '正', '确', '的', '党', '!']
"""

# 3、view dictionary
# print(tokenizer.vocab)

# size of vocabulary
print(tokenizer.vocab_size)
"""
21128
"""

# 4、index transfer
# 4.1 transfer word sequence to index sequence
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
"""
[704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 
3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 
1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106]
"""

# 4.2 transfer index sequence to word sequence
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)
"""
['中', '华', '民', '族', '是', '一', '个', '伟', '大', 
'的', '民', '族', '，', '中', '国', '共', '产', '党', 
'是', '一', '个', '伟', '大', '、', '光', '荣', '、', '正', '确', '的', '党', '!']
"""

# 4.3 transfer tokens to string
str_sen = tokenizer.convert_tokens_to_string(tokens)
print(str_sen)
"""
中 华 民 族 是 一 个 伟 大 的 民 族 ， 中 国 共 产 党 是 一 个 伟 大 、 光 荣 、 正 确 的 党!
"""

# 5、transfer tokens and ids with encode/decode in a more convenient way
# 5.1 encode string->id sequence
ids = tokenizer.encode(sentence, add_special_tokens=True)
print(ids)
"""
[101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106, 102]
"""

# 5.2 decode id sequence -> string
str_sen = tokenizer.decode(ids, skip_special_tokens=True)
print(str_sen)
"""
skip_special_tokens=True
中 华 民 族 是 一 个 伟 大 的 民 族 ， 中 国 共 产 党 是 一 个 伟 大 、 光 荣 、 正 确 的 党!

skip_special_tokens=False
[CLS] 中 华 民 族 是 一 个 伟 大 的 民 族 ， 中 国 共 产 党 是 一 个 伟 大 、 光 荣 、 正 确 的 党! [SEP
"""

# 6. padding and truncate
# 6.1 padding
ids = tokenizer.encode(sentence, padding="max_length", max_length=45)
print(ids)
"""
[101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
# 6.2 truncation
ids = tokenizer.encode(sentence, max_length=15, truncation=True)
print(ids)
"""
[101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 102]
"""

# 7. other input
ids = tokenizer.encode(sentence, padding="max_length", max_length=15)
print(ids)
"""
[101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106, 102]
"""
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids= [0] * len(ids)
print("ids: ", ids, "attention_mask: ", attention_mask, "token_type_ids:", token_type_ids)

# 8. quick method for call
inputs = tokenizer.encode_plus(sentence, padding="max_length", max_length=15)
print(inputs)
"""
{'input_ids': [101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""


inputs = tokenizer(sentence, padding="max_length", max_length=15)
print(inputs)
"""
{'input_ids': [101, 704, 1290, 3696, 3184, 3221, 671, 702, 836, 1920, 4638, 3696, 3184, 8024, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 106, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

# 9. handle batch dataset
sentences = [
    "中华文化渊源流长", 
    "中华民族是个伟大的民族", 
    "中国共产党是一个伟大、光荣、正确的党"
]
res = tokenizer(sentences)
print(res)
"""
{'input_ids': 
[[101, 704, 1290, 3152, 1265, 3929, 3975, 3837, 7270, 102], 
[101, 704, 1290, 3696, 3184, 3221, 702, 836, 1920, 4638, 3696, 3184, 102], 
[101, 704, 1744, 1066, 772, 1054, 3221, 671, 702, 836, 1920, 510, 1045, 5783, 510, 3633, 4802, 4638, 1054, 102]], 
'token_type_ids': 
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
'attention_mask': 
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
"""


# 9.1 handle each sentence cyclically
import time
import torch
times = []
for i in range(1000):
    torch.cuda.synchronize()
    start = time.time()
    tokenizer(sentence)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

print("average time consuming for 100 times singly: ", sum(times))
"""
average time consuming for 100 times singly:  0.0010997796058654786
"""

# 9.2 handle batch sentences
import time
torch.cuda.synchronize()
start = time.time()
tokenizer([sentence]*1000)
torch.cuda.synchronize()
end = time.time()
times = end - start
print("average time consuming for 1000 data batch: ", times)
"""
average time consuming for 1000 data batch:  0.020461559295654297
"""

# 10、fast / slow tokenizer

# 10.1 fast tokenizer
sentence = "python是最简单的编程语言， 也是最难的编程语言"
fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
# print(fast_tokenizer)
"""
BertTokenizerFast(name_or_path='uer/roberta-base-finetuned-dianping-chinese', 
vocab_size=21128, model_max_length=1000000000000000019884624838656, 
is_fast=True, padding_side='right', truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 
'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, 
clean_up_tokenization_spaces=True),  added_tokens_decoder={
0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
"""


# 10.2 slow tokenizer
slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", 
                                               use_fast=False)
print(slow_tokenizer)
"""
BertTokenizer(name_or_path='uer/roberta-base-finetuned-dianping-chinese', 
vocab_size=21128, model_max_length=1000000000000000019884624838656, 
is_fast=False, padding_side='right', truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 
'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, 
clean_up_tokenization_spaces=True),  added_tokens_decoder={
0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
"""

# 10.1 handle each sentence cyclically with fast_tokenizer
import time
import torch
times = []
for i in range(1000):
    torch.cuda.synchronize()
    start = time.time()
    fast_tokenizer(sentence)
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)
print("average time consuming for 100 times with fast_tokenizer: ", sum(times))
"""
average time consuming for 100 times with fast_tokenizer:  0.09179449081420898
"""


# 10.2 handle each sentence cyclically with slow_tokenizer
times_slow = []
for i in range(1000):
    torch.cuda.synchronize()
    start = time.time()
    slow_tokenizer(sentence)
    torch.cuda.synchronize()
    end = time.time()
    times_slow.append(end - start)

print("average time consuming for 100 times with slow_tokenizer: ", sum(times_slow))
"""
average time consuming for 100 times with slow_tokenizer:  0.22511005401611328
"""

# 11. offset_mapping  [tokenizer之后token与原文的对应关系]
# sentence: "python是最简单的编程语言， 也是最难的编程语言"
inputs = fast_tokenizer(sentence, return_offsets_mapping=True)
print(inputs)
"""
{'input_ids': 
[101, 9030, 3221, 3297, 5042, 1296, 4638, 5356, 4923, 6427, 6241, 8024, 738, 3221, 3297, 7410, 4638, 5356, 4923, 6427, 6241, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
'offset_mapping': [(0, 0), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (0, 0)]}
"""
print(inputs.word_ids())
"""
[None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]
"""
# inputs = slow_tokenizer(sentence, return_offsets_mapping=True)
# print(inputs)
"""
NotImplementedError: return_offset_mapping is not available 
when using Python tokenizers. To use this feature, 
change your tokenizer to one deriving from transformers.
PreTrainedTokenizerFast. More information on available 
tokenizers at https://github.com/huggingface/transformers/pull/2674
"""

# 12、 loading specical tokenizers  <--transformers版本不同， 导致无法运行-->
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", 
#                                           trust_remote_code=True)
# print(tokenizer)








