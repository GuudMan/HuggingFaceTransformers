from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
print(tokenizer)