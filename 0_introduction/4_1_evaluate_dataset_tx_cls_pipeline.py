from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

model = AutoModelForSequenceClassification.from_pretrained("./4_tx_cls_model")
model.cuda()
sentence = "这家很差错， 不好吃"
id2_label = {0: "差评", 1: "好评"}
model.eval()
model.config.id2label = id2_label
tokenizer = AutoTokenizer.from_pretrained("./4_tx_cls_model")
pipe = pipeline("text-classification", model=model, 
                tokenizer=tokenizer, 
                device=0)
print(pipe(sentence))








