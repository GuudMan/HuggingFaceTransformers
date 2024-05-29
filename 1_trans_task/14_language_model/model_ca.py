# 1、 import packages
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, BloomForCausalLM

model = AutoModelForCausalLM.from_pretrained("Langboat/bloom-389m-zh")




