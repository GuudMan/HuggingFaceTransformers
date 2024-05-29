
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./rbt3_tx_cls")
tokenizer = AutoTokenizer.from_pretrained("./rbt3_tx_cls")
model.cuda()
infer_sen = "这家太好了"
id2_label = {0: "差评", 1: "好评"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(infer_sen, return_tensors="pt")
    inputs = {k:v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print("预测结果：", id2_label.get(pred.item()))