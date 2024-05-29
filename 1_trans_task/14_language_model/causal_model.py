# 1、 import packages
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BloomForCausalLM

# 2、 load datasets
print('--------# 2、 load datasets------')
ds = Dataset.load_from_disk("./wiki_cn_filtered")
print(ds)
"""
Dataset({
    features: ['source', 'completion'],
    num_rows: 10000
})
"""
print(ds[0])
"""
{'source': 'wikipedia.zh2307', 
'completion': "西安交通大学博物馆（Xi'an Jiaotong University Museum）是一座位于西安交通大学的博物馆，馆长是锺明善。\n历史\n2004年9月20日开始筹建，2013年4月8日正式建成开馆，位于西安交通大学兴庆校区陕西省西安市咸宁西路28号。建筑面积6,800平米，展厅面积4,500平米，馆藏文物4,900余件。包括历代艺术文物馆、碑石书法馆、西部农民画馆、邢良坤陶瓷艺术馆、陕西秦腔博物馆和书画展厅共五馆一厅。\n营业时间\n* 周一至周六：上午九点至十二点，下午一点至五点\n* 周日闭馆"}
"""

# 3、datasets preprocess
print('--------# 3、datasets preprocess------')
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-389m-zh")
def process_func(examples):
    contents = [e + tokenizer.eos_token for e in examples["completion"]]
    return tokenizer(contents, max_length=384, truncation=True)

tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)
print(tokenized_ds)
"""
Dataset({
    features: ['input_ids', 'attention_mask'],
    num_rows: 10000
})
"""
print('--------# 3.1 DataLoader------')
from torch.utils.data import DataLoader
dl = DataLoader(tokenized_ds, batch_size=2, 
                collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
print(next(enumerate(dl)))
print(tokenizer.pad_token)
"<pad>"
print(tokenizer.pad_token_id)
"3"
print(tokenizer.eos_token)
"</s>"
print(tokenizer.eos_token_id)
"2"
# 4、create model
print('--------# 4、create model------')
model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")


# 5、configure TrainingArguments
print('--------# 5、configure TrainingArguments------')
args = TrainingArguments(
    output_dir="./causal_lm", 
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=8, 
    gradient_accumulation_steps=8, 
    logging_steps=10, 
    num_train_epochs=3, 
    # save_strategy="epoch",  
    # load_best_model_at_end=True, 
    save_total_limit=1 
)

# 8、configure Trainer
print('--------# 6、configure Trainer------')
trainer = Trainer(args=args, 
                  model=model, 
                  train_dataset=tokenized_ds, 
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))



# 9、model train
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:3950

print('--------# 7、model train------')
trainer.train()
model.save_pretrained("./causal_model")


from transformers import pipeline
# 10、model infer

pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=0
                )
res = pipe("下面是来自北京电视台的最新报道。今日， 北京市气象局发布最新提醒， 新一轮寒潮即将来袭， 请广大市民", max_length=128, do_sample=True)
print(res)



