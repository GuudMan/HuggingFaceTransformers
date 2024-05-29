from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
model = AutoModelForTokenClassification.from_pretrained("./models_for_ner/checkpoint-738")
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
model.config

if torch.cuda.is_available():
    model = model.cuda()

tokenizer = AutoTokenizer.from_pretrained("/home/oem/pn/01_transformers/1_trans_task/08_trans_so/tx_cls_transf_token")

sentences = ""

model.eval()
with torch.inference_mode():

    inputs = tokenizer(sentences, return_tensors="pt")
    inputs = {k:v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs).logits
    print(outputs)
    pred = torch.argmax(outputs, dim=0)
    print(pred)