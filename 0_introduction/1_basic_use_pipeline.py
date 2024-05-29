from transformers.pipelines import SUPPORTED_TASKS
print(SUPPORTED_TASKS.items())
"""
dict_items([('audio-classification', {'impl': <class 'transformers.pipelines.audio_classification.AudioClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForAudioClassification'>,), 'default': {'model': {'pt': ('superb/wav2vec2-base-superb-ks', '372e048')}}, 'type': 'audio'}), ('automatic-speech-recognition', {'impl': <class 'transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForCTC'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSpeechSeq2Seq'>), 'default': {'model': {'pt': ('facebook/wav2vec2-base-960h', '55bb623')}}, 'type': 'multimodal'}), ('text-to-audio', {'impl': <class 'transformers.pipelines.text_to_audio.TextToAudioPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTextToWaveform'>, <class 'transformers.models.auto.modeling_auto.AutoModelForTextToSpectrogram'>), 'default': {'model': {'pt': ('suno/bark-small', '645cfba')}}, 'type': 'text'}), ('feature-extraction', {'impl': <class 'transformers.pipelines.feature_extraction.FeatureExtractionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModel'>,), 'default': {'model': {'pt': ('distilbert-base-cased', '935ac13'), 'tf': ('distilbert-base-cased', '935ac13')}}, 'type': 'multimodal'}), ('text-classification', {'impl': <class 'transformers.pipelines.text_classification.TextClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSequenceClassification'>,), 'default': {'model': {'pt': ('distilbert-base-uncased-finetuned-sst-2-english', 'af0f99b'), 'tf': ('distilbert-base-uncased-finetuned-sst-2-english', 'af0f99b')}}, 'type': 'text'}), ('token-classification', {'impl': <class 'transformers.pipelines.token_classification.TokenClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTokenClassification'>,), 'default': {'model': {'pt': ('dbmdz/bert-large-cased-finetuned-conll03-english', 'f2482bf'), 'tf': ('dbmdz/bert-large-cased-finetuned-conll03-english', 'f2482bf')}}, 'type': 'text'}), ('question-answering', {'impl': <class 'transformers.pipelines.question_answering.QuestionAnsweringPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForQuestionAnswering'>,), 'default': {'model': {'pt': ('distilbert-base-cased-distilled-squad', '626af31'), 'tf': ('distilbert-base-cased-distilled-squad', '626af31')}}, 'type': 'text'}), ('table-question-answering', {'impl': <class 'transformers.pipelines.table_question_answering.TableQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForTableQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('google/tapas-base-finetuned-wtq', '69ceee2'), 'tf': ('google/tapas-base-finetuned-wtq', '69ceee2')}}, 'type': 'text'}), ('visual-question-answering', {'impl': <class 'transformers.pipelines.visual_question_answering.VisualQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVisualQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('dandelin/vilt-b32-finetuned-vqa', '4355f59')}}, 'type': 'multimodal'}), ('document-question-answering', {'impl': <class 'transformers.pipelines.document_question_answering.DocumentQuestionAnsweringPipeline'>, 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForDocumentQuestionAnswering'>,), 'tf': (), 'default': {'model': {'pt': ('impira/layoutlm-document-qa', '52e01b3')}}, 'type': 'multimodal'}), ('fill-mask', {'impl': <class 'transformers.pipelines.fill_mask.FillMaskPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForMaskedLM'>,), 'default': {'model': {'pt': ('distilroberta-base', 'ec58a5b'), 'tf': ('distilroberta-base', 'ec58a5b')}}, 'type': 'text'}), ('summarization', {'impl': <class 'transformers.pipelines.text2text_generation.SummarizationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {'model': {'pt': ('sshleifer/distilbart-cnn-12-6', 'a4f8f3e'), 'tf': ('t5-small', 'd769bba')}}, 'type': 'text'}), ('translation', {'impl': <class 'transformers.pipelines.text2text_generation.TranslationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {('en', 'fr'): {'model': {'pt': ('t5-base', '686f1db'), 'tf': ('t5-base', '686f1db')}}, ('en', 'de'): {'model': {'pt': ('t5-base', '686f1db'), 'tf': ('t5-base', '686f1db')}}, ('en', 'ro'): {'model': {'pt': ('t5-base', '686f1db'), 'tf': ('t5-base', '686f1db')}}}, 'type': 'text'}), ('text2text-generation', {'impl': <class 'transformers.pipelines.text2text_generation.Text2TextGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>,), 'default': {'model': {'pt': ('t5-base', '686f1db'), 'tf': ('t5-base', '686f1db')}}, 'type': 'text'}), ('text-generation', {'impl': <class 'transformers.pipelines.text_generation.TextGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>,), 'default': {'model': {'pt': ('gpt2', '6c0e608'), 'tf': ('gpt2', '6c0e608')}}, 'type': 'text'}), ('zero-shot-classification', {'impl': <class 'transformers.pipelines.zero_shot_classification.ZeroShotClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSequenceClassification'>,), 'default': {'model': {'pt': ('facebook/bart-large-mnli', 'c626438'), 'tf': ('roberta-large-mnli', '130fb28')}, 'config': {'pt': ('facebook/bart-large-mnli', 'c626438'), 'tf': ('roberta-large-mnli', '130fb28')}}, 'type': 'text'}), ('zero-shot-image-classification', {'impl': <class 'transformers.pipelines.zero_shot_image_classification.ZeroShotImageClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForZeroShotImageClassification'>,), 'default': {'model': {'pt': ('openai/clip-vit-base-patch32', 'f4881ba'), 'tf': ('openai/clip-vit-base-patch32', 'f4881ba')}}, 'type': 'multimodal'}), ('zero-shot-audio-classification', {'impl': <class 'transformers.pipelines.zero_shot_audio_classification.ZeroShotAudioClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModel'>,), 'default': {'model': {'pt': ('laion/clap-htsat-fused', '973b6e5')}}, 'type': 'multimodal'}), ('conversational', {'impl': <class 'transformers.pipelines.conversational.ConversationalPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForSeq2SeqLM'>, <class 'transformers.models.auto.modeling_auto.AutoModelForCausalLM'>), 'default': {'model': {'pt': ('microsoft/DialoGPT-medium', '8bada3b'), 'tf': ('microsoft/DialoGPT-medium', '8bada3b')}}, 'type': 'text'}), ('image-classification', {'impl': <class 'transformers.pipelines.image_classification.ImageClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageClassification'>,), 'default': {'model': {'pt': ('google/vit-base-patch16-224', '5dca96d'), 'tf': ('google/vit-base-patch16-224', '5dca96d')}}, 'type': 'image'}), ('image-segmentation', {'impl': <class 'transformers.pipelines.image_segmentation.ImageSegmentationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageSegmentation'>, <class 'transformers.models.auto.modeling_auto.AutoModelForSemanticSegmentation'>), 'default': {'model': {'pt': ('facebook/detr-resnet-50-panoptic', 'fc15262')}}, 'type': 'multimodal'}), ('image-to-text', {'impl': <class 'transformers.pipelines.image_to_text.ImageToTextPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVision2Seq'>,), 'default': {'model': {'pt': ('ydshieh/vit-gpt2-coco-en', '65636df'), 'tf': ('ydshieh/vit-gpt2-coco-en', '65636df')}}, 'type': 'multimodal'}), ('object-detection', {'impl': <class 'transformers.pipelines.object_detection.ObjectDetectionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForObjectDetection'>,), 'default': {'model': {'pt': ('facebook/detr-resnet-50', '2729413')}}, 'type': 'multimodal'}), ('zero-shot-object-detection', {'impl': <class 'transformers.pipelines.zero_shot_object_detection.ZeroShotObjectDetectionPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForZeroShotObjectDetection'>,), 'default': {'model': {'pt': ('google/owlvit-base-patch32', '17740e1')}}, 'type': 'multimodal'}), ('depth-estimation', {'impl': <class 'transformers.pipelines.depth_estimation.DepthEstimationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForDepthEstimation'>,), 'default': {'model': {'pt': ('Intel/dpt-large', 'e93beec')}}, 'type': 'image'}), ('video-classification', {'impl': <class 'transformers.pipelines.video_classification.VideoClassificationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForVideoClassification'>,), 'default': {'model': {'pt': ('MCG-NJU/videomae-base-finetuned-kinetics', '4800870')}}, 'type': 'video'}), ('mask-generation', {'impl': <class 'transformers.pipelines.mask_generation.MaskGenerationPipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForMaskGeneration'>,), 'default': {'model': {'pt': ('facebook/sam-vit-huge', '997b15')}}, 'type': 'multimodal'}), ('image-to-image', {'impl': <class 'transformers.pipelines.image_to_image.ImageToImagePipeline'>, 'tf': (), 'pt': (<class 'transformers.models.auto.modeling_auto.AutoModelForImageToImage'>,), 'default': {'model': {'pt': ('caidas/swin2SR-classical-sr-x2-64', '4aaedcb')}}, 'type': 'image'})])
"""

for k, v in SUPPORTED_TASKS.items():
    print(k, v)

#### 1. creat and use for pipeline
from transformers import pipeline
pipe = pipeline('text-classification')
#### 2. use pipeline to infer
print(pipe(['I think it is very nice', 'oh!, so bad!']))
"""
[{'label': 'POSITIVE', 'score': 0.9998282194137573}, {'label': 'NEGATIVE', 'score': 0.999769389629364}]
"""

# 3. Specifying inference types and models
pipe = pipeline('text-classification', 
                model="uer/roberta-base-finetuned-dianping-chinese")
print(pipe("我觉得这个方法不太行！"))
"""
[{'label': 'negative (stars 1, 2 and 3)', 'score': 0.9742688536643982}]
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
# 4. Preloading model then create Pipeline
# 4.1 Specify model and tokenizer simutaneously
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-dianping-chinese')
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(pipe("这个城市景色优美"))
"""
[{'label': 'positive (stars 4 and 5)', 'score': 0.9255221486091614}]
"""

# 4.2 check runing device
print(pipe.model.device)
"""
cpu
"""

# 4.3 Statistical time consuming for 100 inference
import torch
import time
times = []
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    pipe("我比较喜欢这个")
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

print("device: ", pipe.model.device, "average time consuming for 100 times: ", sum(times) / 100)
"""
device:  cpu average time consuming for 100 times:  0.01846954822540283
"""


# 4.4 use cuda to inference
pipe = pipeline("text-classification", 
                model="uer/roberta-base-finetuned-dianping-chinese", device=1)

# statistic time consuming in cuda for 100 inference
import torch
import time
times = []
for i in range(100):
    torch.cuda.synchronize()
    start = time.time()
    pipe("我比较喜欢这个")
    torch.cuda.synchronize()
    end = time.time()
    times.append(end - start)

print("device: ", pipe.model.device, "average time consuming for 100 times: ", sum(times) / 100)
"""
device:  cuda:0 average time consuming for 100 times:  0.011090264320373536
"""

# 5、Parameters for pipeline
qa_pipe = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)
from transformers import QuestionAnsweringPipeline
print(qa_pipe(question="中国的最美城市在哪里？", context="上海是中国最美城市"))
"""
{'score': 0.8211117386817932, 'start': 0, 'end': 2, 'answer': '上海'}
"""

# 6、transformers for object detection   
det_pipe = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection", device=0)

# 6.1 image inference
import requests
from PIL import Image
import matplotlib.pyplot as plt
url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MTR8fHBpY25pY3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
im = Image.open(requests.get(url, stream=True).raw)
# im.show()
plt.imshow(im)
plt.show()
predictions = det_pipe(im, candidate_labels=['hat', 'human'])
print(predictions)
""" Outputs only the targets that are already in the image
[{'score': 0.10808248817920685, 'label': 'hat', 'box': {'xmin': 38, 'ymin': 172, 'xmax': 260, 'ymax': 363}}]
"""

from PIL import ImageDraw
draw = ImageDraw.Draw(im)
for prediction in predictions:
    box = prediction["box"]
    label = prediction['label']
    score = prediction['score']
    xmin, ymin, xmax, ymax = box.values()
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}:      {round(score, 2)}", fill="red")
im.save("./transformers_object_detect.jpg")

# 7. Underlying principle
from transformers import *
import torch

tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

input_text = "这个还不错！"
inputs = tokenizer(input_text, return_tensors="pt")
print(inputs)
"""
{'input_ids': tensor([[ 101, 6821,  702, 6820,  679, 7231, 8013,  102]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
"""

res = model(**inputs)
print(res)
"""
SequenceClassifierOutput(loss=None, logits=tensor([[-0.2160,  0.1355]], 
grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
logits = res.logits
logits = torch.softmax(logits, dim=-1)
print(logits)
"""
tensor([[0.4130, 0.5870]], grad_fn=<SoftmaxBackward0>)
"""

pred = torch.argmax(logits).item()
print(pred)
"""
1
"""

print(model.config.id2label)
"""
{0: 'negative (stars 1, 2 and 3)', 1: 'positive (stars 4 and 5)'}
"""

result = model.config.id2label.get(pred)
print(result)
"""
positive (stars 4 and 5)
"""


