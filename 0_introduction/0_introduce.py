import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gradio as gr
from transformers import *
# 通过Interface加载pipeline并启动questionAnswering服务
gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()
