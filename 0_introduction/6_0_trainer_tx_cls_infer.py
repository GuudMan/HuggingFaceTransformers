from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./6_0_trainer_tx_cls/checkpoint-78")
tokenizer = AutoTokenizer.from_pretrained("./6_trainer_tx_cls")
model = model.cuda() if torch.cuda.is_available() else model
sentences = "我错"
model.eval()

id2label = {0:"差评", 1:"好评"}
import torch
with torch.inference_mode():
    inputs = tokenizer(sentences, return_tensors="pt")
    inputs = {k:v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs).logits
    pred = torch.argmax(outputs, dim=-1)
    print("推理结果：", id2label.get(pred.item()))
