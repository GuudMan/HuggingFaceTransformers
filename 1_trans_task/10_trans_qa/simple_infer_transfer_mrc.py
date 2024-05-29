from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model = AutoModelForQuestionAnswering.from_pretrained("./model_for_qa/checkpoint-120")
tokenizer = AutoTokenizer.from_pretrained("/home/oem/pn/01_transformers/1_trans_task/08_trans_so/tx_cls_transf_token")

pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)
res = pipe(question="清澈的爱？", context="清澈的爱， 名为中国。")
print(res)
