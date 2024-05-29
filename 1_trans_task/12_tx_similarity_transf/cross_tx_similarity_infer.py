from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("/home/oem/pn/01_transformers/1_trans_task/08_trans_so/tx_cls_transf_token")
model = AutoModelForSequenceClassification.from_pretrained("./cross_model_tx_similarity/checkpoint-96")
model.config.id2label = {0:"不相似", 1:"相似"}
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
result = pipe({"text": "爱我中华", "text_pair":"中华爱我"}, function_to_apply="none")
result['label'] = "相似" if result['score'] > 0.5 else "不相似"
print(result)















