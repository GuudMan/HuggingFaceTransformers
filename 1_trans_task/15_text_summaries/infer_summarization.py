from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from summarization_transf import ds
model = AutoModelForSeq2SeqLM.from_pretrained("./summary/checkpoint-114")
tokenizer = AutoTokenizer.from_pretrained("/home/oem/pn/01_transformers/1_trans_task/08_trans_so/tx_cls_transf_token")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
res = pipe("摘要生成: \n" + ds['test'][-1]['content'], max_length=64, do_sample=True)
print(res)