from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# load model from local
model = AutoModelForSequenceClassification.from_pretrained("./5_tx_cls_metrics")
model = model.cuda() if torch.cuda.is_available() else model
tokenizer = AutoTokenizer.from_pretrained("./5_tx_cls_metrics")

sentence = "不错"
model.eval()
id2label = {0:"差评", 1:"好评"}
import torch
with torch.inference_mode():
    inputs = tokenizer(sentence, 
                             max_length=128, 
                             truncation=True, 
                             return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    # outputs = model(**inputs)  SequenceClassifierOutput(loss=None, logits=tensor([[-3.8990,  3.7123]], device='cuda:0'), hidden_states=None, attentions=None)
    outputs = model(**inputs).logits
    print(outputs)  # tensor([[-3.8990,  3.7123]], device='cuda:0'
    pred = torch.argmax(outputs, dim=-1)  # tensor([1], device='cuda:0')
    print(pred)
    pred_cls = id2label.get((pred.item()))
    print("推理结果：", pred_cls)

# use pipeline to inference
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
res = pipe(sentence)
print(res)



    