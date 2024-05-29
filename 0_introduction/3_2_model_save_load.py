from transformers import AutoConfig, AutoModel, AutoTokenizer

# 1、 load model
# 1.1 load online
# model = AutoModel.from_pretrained("hfl/rbt3", force_download=True)

# 1.2 load offline
model = AutoModel.from_pretrained("hfl/rbt3")
print(model.config)
"""
BertConfig {
  "_name_or_path": "hfl/rbt3",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.36.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
"""

# 2、 model call
sen = "美丽中国"
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
inputs = tokenizer(sen, return_tensors="pt")
print(inputs)
"""
{'input_ids': tensor([[ 101, 5401,  714,  704, 1744,  102]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
"""

# 2.1 without model head
model = AutoModel.from_pretrained("hfl/rbt3", output_attentions=True)
output = model(**inputs)
# print(output)

print(output.last_hidden_state.size())
"""
torch.Size([1, 6, 768])
"""

print(len(inputs["input_ids"][0]))
"""
6
"""

# 2.2 with model
from transformers import AutoModelForSequenceClassification
clz_model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=10)
outputs = clz_model(**inputs)
print(outputs)
"""
SequenceClassifierOutput(loss=None, logits=tensor([[ 0.0301,  0.7696, -0.2856,  0.6312,  0.0484, -0.0517,  0.2084, -0.0107,
         -0.2997, -0.5594]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
num_labels = clz_model.config.num_labels
print(num_labels)
"""
10
"""
