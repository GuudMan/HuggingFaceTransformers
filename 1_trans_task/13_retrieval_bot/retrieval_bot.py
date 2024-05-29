# 1、 import packages

# 2、 load datasets
print('--------# 2、 load datasets------')
import pandas as pd
data = pd.read_csv('./law_faq.csv')
print(data.head())
print(data.columns)
"""
Index(['title', 'reply'], dtype='object')
"""

# 2、load model
print('--------# 2、load model------')
from dual_model import DualModel
dual_model = DualModel.from_pretrained("../12_tx_similarity_transf/dual_model/checkpoint-189")
dual_model = dual_model.cuda()
dual_model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")


# 3、 encode text to vector
print('--------# 6、create evaluate function------')
import torch
from tqdm import tqdm 
questions = data['title'].to_list()
vectors = []
with torch.inference_mode():
    for i in tqdm(range(0, len(questions), 32)):
        batch_sens = questions[i: i + 32]
        inputs = tokenizer(batch_sens, return_tensors="pt", padding=True, 
                           max_length=128, truncation=True)
        inputs = {k:v.to(dual_model.device) for k, v in inputs.items()}
        vector = dual_model.bert(**inputs)[1]
        vectors.append(vector)
vectors = torch.concat(vectors, dim=0).cpu().numpy()
print(vectors.shape)
"""
(18213, 768)
"""

# 4、create index
print("----------4、create index------------")
import faiss
index = faiss.IndexFlatIP(768)
faiss.normalize_L2(vectors)
index.add(vectors)
print(index)

# 5、question text encode
print("----------# 5、question text encode------------")
question = "恶意诽谤"
with torch.inference_mode():
    inputs = tokenizer(question, 
                       return_tensors="pt", padding=True, 
                       max_length=128, truncation=True)
    inputs = {k:v.to(dual_model.device) for k, v in inputs.items()}
    vector = dual_model.bert(**inputs)[1]
    q_vector = vector.cpu().numpy()
print(q_vector.shape)
"""
(1, 768)
"""

# 6、vector match(recall)
print('--------# 6、vector match(recall)------')
faiss.normalize_L2(q_vector)
scores, indexes = index.search(q_vector, 10)
topk_result = data.values[indexes[0].tolist()]
# print(topk_result)
print(topk_result[:, 0])


# 7、load model
print('--------# 7、load model------')
from transformers import BertForSequenceClassification
cross_model = BertForSequenceClassification.from_pretrained("../12_tx_similarity_transf/cross_model_tx_similarity/checkpoint-96")
cross_model = cross_model.cuda()
cross_model.eval()

# 8、model prediction(sorted)
print('--------# 8、model prediction(sorted)------')
candidate = topk_result[:, 0].tolist()
ques = [question] * len(candidate)
inputs = tokenizer(ques, candidate, return_tensors="pt", padding=True, max_length=128, truncation=True)
inputs = {k: v.to(cross_model.device) for k, v in inputs.items()}
with torch.inference_mode():
    logits = cross_model(**inputs).logits.squeeze()
    result = torch.argmax(logits, dim=-1)

print(result)

candidate_answer = topk_result[:, 1].tolist()
match_question = candidate[result.item()]
final_answer = candidate_answer[result.item()]
print(match_question)
print(final_answer)




