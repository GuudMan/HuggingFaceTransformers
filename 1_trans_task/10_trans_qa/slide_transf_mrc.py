# 1、 import package

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict

# 2、load datasets
datasets = DatasetDict.load_from_disk("./mrc_data")
print('-----------# 2、load datasets----------')
print(datasets)
print(datasets['train'][0])

# 3、datasets split
print('-----------# 3、datasets split----------')
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
                               return_overflowing_tokens=True, 
                               stride=128, 
                               max_length=384, 
                               truncation="only_second", 
                               padding="max_length"
                               )
print("-----------# 3.1 tokenize examples----------")
print(tokenized_examples.keys())
"""
dict_keys(['input_ids', 'token_type_ids', 
'attention_mask', 'offset_mapping', 
'overflow_to_sample_mapping'])
"""
# print(tokenized_examples['input_ids'])
print(tokenized_examples['overflow_to_sample_mapping'],
       len(tokenized_examples['overflow_to_sample_mapping']))
"""
[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9] 29
"""

for sen in tokenizer.batch_decode(tokenized_examples['input_ids'][:3]):
    print(sen)
print("-----------# 3.2 offset_mapping----------")
# print(tokenized_examples['offset_mapping'][:3])
# print(tokenized_examples['offset_mapping'][0], 
#       len(tokenized_examples['offset_mapping'][0]))
sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
for idx, _ in enumerate(sample_mapping):
    answer = sample_dataset['answers'][sample_mapping[idx]]
    start_char = answer['answer_start'][0]
    end_char = start_char + len(answer['text'][0])

    # 定位答案在token中的起始和结束位置
    # 拿到context的起始和结束， 然后从左右两侧逼近

    context_start = tokenized_examples.sequence_ids(idx).index(1)
    context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1

    offset = tokenized_examples.get("offset_mapping")[idx]
    example_ids = []

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
        # example_ids.append(examples["id"][sample_mapping[idx]])
    print(answer, start_char, end_char, context_start, context_end, 
          start_token_pos, end_token_pos)
    print("token answer decode:", 
          tokenizer.decode(tokenized_examples['input_ids'][idx][start_token_pos:end_token_pos + 1]))

def process_func(examples):
    tokenized_examples = tokenizer(text=examples['question'], 
                                   text_pair=examples['context'], 
                                   max_length=384, 
                                   stride=128, 
                                   return_offsets_mapping=True, 
                                   return_overflowing_tokens=True, 
                                   truncation="only_second", 
                                   padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    start_positions = []
    end_positions = []
    example_ids = []
    for idx, _ in enumerate(sample_mapping):
        # amswer: {'text':str, answer_start: int}
        answer = examples['answers'][sample_mapping[idx]]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])

        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1

        offset = tokenized_examples.get("offset_mapping")[idx]

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
        example_ids.append(examples['id'][sample_mapping[idx]])
        tokenized_examples['offset_mapping'][idx] = [
            (o if tokenized_examples.sequence_ids(idx)[k] == 1 else None) 
            for k, o in enumerate(tokenized_examples['offset_mapping'][idx])
        ]
    tokenized_examples['example_ids'] = example_ids
    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    return tokenized_examples
  
tokenized_datasets = datasets.map(process_func, batched=True, 
                                  remove_columns=datasets['train'].column_names)
print(tokenized_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_ids', 'start_positions', 'end_positions'],
        num_rows: 19189
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_ids', 'start_positions', 'end_positions'],
        num_rows: 6327
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_ids', 'start_positions', 'end_positions'],
        num_rows: 1988
    })
})
"""
# print(tokenized_datasets['train'][0])
print('-------3.3 train---offset_mapping-------')
# print(tokenized_datasets['train']['offset_mapping'][1])
"""
[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, [266, 267], [267, 268], [268, 269], [269, 270], [270, 271], [271, 272], [272, 273], [273, 274], [274, 275], [275, 276], [276, 277], [277, 278], [278, 279], [279, 280], [280, 281], [281, 282], [282, 283], [283, 284], [284, 285], [285, 286], [286, 287], [287, 288], [288, 289], [289, 290], [290, 291], [291, 292], [292, 293], [293, 294], [294, 295], [295, 296], [296, 297], [297, 298], [298, 299], [299, 300], [300, 301], [301, 302], [302, 303], [303, 304], [304, 305], [305, 306], [306, 307], [307, 308], [308, 309], [309, 310], [310, 311], [311, 312], [312, 313], [313, 314], [314, 315], [315, 316], [316, 317], [317, 318], [318, 319], [319, 320], [320, 321], [321, 325], [325, 326], [326, 327], [327, 328], [328, 329], [329, 330], [330, 331], [331, 332], [332, 333], [333, 334], [334, 335], [335, 336], [336, 337], [337, 338], [338, 339], [339, 340], [340, 341], [341, 342], [342, 343], [343, 344], [344, 345], [345, 346], [346, 347], [347, 348], [348, 349], [349, 350], [350, 351], [351, 352], [352, 353], [353, 354], [354, 355], [355, 356], [356, 360], [360, 361], [361, 362], [362, 363], [363, 364], [364, 365], [365, 366], [366, 367], [367, 368], [368, 369], [369, 370], [370, 371], [371, 372], [372, 373], [373, 374], [374, 375], [375, 376], [376, 377], [377, 378], [378, 379], [379, 380], [380, 381], [381, 382], [382, 383], [383, 384], [384, 385], [385, 386], [386, 387], [387, 388], [388, 390], [390, 391], [391, 392], [392, 393], [393, 394], [394, 395], [395, 396], [396, 397], [397, 398], [398, 399], [399, 400], [400, 401], [401, 402], [402, 403], [403, 404], [404, 405], [405, 406], [406, 407], [407, 408], [408, 409], [409, 410], [410, 411], [411, 412], [412, 413], [413, 414], [414, 415], [415, 416], [416, 417], [417, 418], [418, 419], [419, 420], [420, 421], [421, 422], [422, 424], [424, 425], [425, 426], [426, 427], [427, 428], [428, 429], [429, 430], [430, 431], [431, 432], [432, 433], [433, 434], [434, 435], [435, 436], [436, 437], [437, 438], [438, 439], [439, 440], [440, 441], [441, 442], [442, 443], [443, 444], [444, 445], [445, 446], [446, 447], [447, 448], [448, 449], [449, 450], [450, 451], [451, 452], [452, 453], [453, 454], [454, 455], [455, 456], [456, 457], [457, 458], [458, 459], [459, 460], [460, 461], [461, 462], [462, 463], [463, 464], [464, 465], [465, 466], [466, 467], [467, 468], [468, 469], [469, 470], [470, 471], [471, 472], [472, 473], [473, 474], [474, 475], [475, 476], [476, 477], [477, 478], [478, 479], [479, 480], [480, 481], [481, 482], [482, 483], [483, 484], [484, 485], [485, 486], [486, 487], [487, 488], [488, 489], [489, 490], [490, 491], [491, 492], [492, 493], [493, 494], [494, 495], [495, 499], [499, 500], [500, 501], [501, 502], [502, 503], [503, 504], [504, 505], [505, 506], [506, 507], [507, 508], [508, 509], [509, 510], [510, 511], [511, 512], [512, 513], [513, 514], [514, 516], [516, 517], [517, 518], [518, 519], [519, 520], [520, 521], [521, 522], [522, 523], [523, 524], [524, 525], [525, 526], [526, 527], [527, 528], [528, 529], [529, 530], [530, 531], [531, 532], [532, 533], [533, 534], [534, 535], [535, 536], [536, 537], [537, 538], [538, 539], [539, 540], [540, 541], [541, 542], [542, 543], [543, 544], [544, 545], [545, 546], [546, 547], [547, 548], [548, 552], [552, 553], [553, 554], [554, 555], [555, 557], [557, 558], [558, 559], [559, 560], [560, 561], [561, 562], [562, 563], [563, 564], [564, 565], [565, 566], [566, 567], [567, 568], [568, 569], [569, 570], [570, 571], [571, 572], [572, 573], [573, 574], [574, 575], [575, 576], [576, 577], [577, 578], [578, 579], [579, 580], [580, 581], [581, 582], [582, 583], [583, 584], [584, 585], [585, 586], [586, 587], [587, 588], [588, 589], [589, 590], [590, 591], [591, 592], [592, 593], [593, 594], [594, 595], [595, 596], [596, 597], [597, 598], [598, 599], [599, 600], [600, 601], [601, 603], [603, 604], [604, 606], [606, 607], [607, 608], [608, 609], [609, 610], [610, 611], [611, 612], [612, 613], [613, 614], [614, 615], [615, 616], [616, 617], [617, 618], [618, 619], [619, 620], [620, 621], [621, 622], [622, 623], [623, 624], [624, 625], [625, 626], [626, 627], [627, 631], [631, 632], [632, 633], [633, 637], [637, 638], [638, 639], [639, 640], [640, 641], [641, 642], [642, 643], [643, 644], [644, 645], [645, 646], [646, 647], [647, 648], [648, 649], [649, 650], [650, 651], [651, 652], [652, 653], [653, 657], [657, 658], [658, 659], None]
"""
# print(tokenized_datasets['train']['example_ids'][:10])
"""
['TRAIN_186_QUERY_0', 'TRAIN_186_QUERY_0', 'TRAIN_186_QUERY_0', 'TRAIN_186_QUERY_1', 'TRAIN_186_QUERY_1', 'TRAIN_186_QUERY_1', 'TRAIN_186_QUERY_2', 'TRAIN_186_QUERY_2', 'TRAIN_186_QUERY_2', 'TRAIN_186_QUERY_3']
"""
# import collections
# # example 和 feature的映射
# example_to_feature = collections.defaultdict(list)
# for idx, example_id in enumerate(tokenized_datasets['train']['example_ids'][:10]):
#     example_to_feature[example_id].append(idx)
# print(example_to_feature)
"""
defaultdict(<class 'list'>, {'TRAIN_186_QUERY_0': [0, 1, 2], 'TRAIN_186_QUERY_1': [3, 4, 5], 'TRAIN_186_QUERY_2': [6, 7, 8], 'TRAIN_186_QUERY_3': [9]})
"""


# 4、get model output
import collections
import numpy as np
def get_result(start_logits, end_logits, examples, features):
    predictions = {}
    references = {}
    # example和feature的映射
    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features['example_ids']):
        example_to_feature[example_id].append(idx)
    
    # 最优答案候选20
    n_best = 20
    # 最大答案长度
    max_ansert_length = 30

    for example in examples:
        example_id = example['id']
        context = example['context']
        answers = []
        for features_idx in example_to_feature[example_id]:
            start_logit = start_logits[features_idx]
            end_logit = end_logits[features_idx]
            offset = features[features_idx]['offset_mapping']
            start_indexes = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexes = np.argsort(end_logit)[::-1][:n_best].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_ansert_length:
                        continue
                    answers.append({
                        "text": context[offset[start_index][0]: offset[end_index][1]], 
                        "score": start_logit[start_index] + end_logit[end_index]
                    })
    
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x['score'])
            predictions[example_id] = best_answer['text']
        else:
            predictions[example_id] = ""
        references[example_id] = example['answers']['text']
    return predictions, references
    


# 5、evaluate functions
from cmrc_eval import evaluate_cmrc
def metric(pred):
    start_logits, end_logits = pred[0]
    if start_logits.shape[0] == len(tokenized_datasets['validation']):
        p, r = get_result(start_logits, end_logits, 
                          datasets['validation'],  
                          tokenized_datasets['validation'])
    else:
        p, r = get_result(start_logits, end_logits, datasets['test'], 
                          tokenized_datasets['test'])
    return evaluate_cmrc(p, r)
  

# 6、load model
model = AutoModelForQuestionAnswering.from_pretrained("hfl/chinese-macbert-base")


# 7、Configure TrainingArguments
args = TrainingArguments(
    output_dir="models_for_qa_slide", 
    per_device_train_batch_size=64, 
    per_device_eval_batch_size=128, 
    evaluation_strategy="steps", 
    eval_steps=200, 
    save_strategy="epoch", 
    logging_steps=50, 
    num_train_epochs=3
)


# 8、configure Trainer
trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_datasets['train'], 
    eval_dataset=tokenized_datasets['validation'], 
    data_collator=DefaultDataCollator(), 
    compute_metrics=metric
)

# 9、train
trainer.train()

# 10、predict
from transformers import pipeline
pipe = pipeline("question-answering", model=model,
                tokenizer=tokenizer, 
                device=0)
res = pipe(question="小明与小光什么关系", context="小明的亲舅妈是小光的大姨夫的亲妹妹")
print(res)









