from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("./4_tx_cls_model")
model.cuda()
sentence = "这家很差错， 不好吃"
id2_label = {0: "差评", 1: "好评"}
model.eval()
tokenizer = AutoTokenizer.from_pretrained("./4_tx_cls_model")
with torch.inference_mode():
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print("结果：", id2_label.get(pred.item()))








