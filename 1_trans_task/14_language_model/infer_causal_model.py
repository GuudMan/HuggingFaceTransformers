from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BloomForCausalLM
# from masked_lm import model
model = AutoModelForCausalLM.from_pretrained("./causal_model/")
tokenizer = AutoTokenizer.from_pretrained("/home/oem/pn/01_transformers/1_trans_task/08_trans_so/tx_cls_transf_token")

# 10、model prediction
print('--------# 10、model prediction------')
from transformers import pipeline
pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device=0)
res = pipe("下面是来自北京电视台的最新报道。今日， 北京市气象局发布最新提醒， 新一轮寒潮即将来袭， 请广大市民", 
           max_length=128, do_sample=True)
print(res)