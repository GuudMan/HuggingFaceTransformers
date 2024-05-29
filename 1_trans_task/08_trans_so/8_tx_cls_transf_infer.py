from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

sentence = "我很喜欢这家"

model = AutoModelForSequenceClassification.from_pretrained("./tx_cls_transf/checkpoint-9")
print(model)
if torch.cuda.is_available():
    model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained("./tx_cls_transf_token")

id2label = {0:"差评", 1:"好评"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs).logits
    pred = torch.argmax(outputs, dim=-1)
    print("推理结果：", id2label.get(pred.item()))




