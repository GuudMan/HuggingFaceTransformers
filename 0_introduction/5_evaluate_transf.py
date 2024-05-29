import evaluate

# 1、 view supported function
print("------# 1、 view supported function------")
print(evaluate.list_evaluation_modules(
    include_community=False, 
    with_details=True))

# 1.1、 accuracy descriptions
print("------# 1.1、 accuracy descriptions------")
accuracy = evaluate.load("accuracy")
print(accuracy.description)
"""
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""

# 2、 view inputs
print("------# 2、 view inputs------")
print(accuracy.inputs_description)
"""
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.
"""

# 2、evaluate indicator caculation-global
print("-----------# 2、evaluate indicator caculation-global---------------")
accuracy = evaluate.load("accuracy")
results = accuracy.compute(references=[0, 1, 2, 0, 1, 2], 
                           predictions=[0, 1, 1, 2, 1, 0])
print(results)
"""
{'accuracy': 0.5}
"""

# 3、evaluate indicator caculation-iteration
print("-----------# 3、evaluate indicator caculation-iteration---0------------")
accuracy = evaluate.load("accuracy")
for ref, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
    accuracy.add(reference=ref, prediction=pred)
res = accuracy.compute()
print(res)
"{'accuracy': 0.5}"

print("-----------# 3、evaluate indicator caculation-iteration---1------------")
accuracy = evaluate.load("accuracy")
for refs, preds in zip([[0, 1], [0, 1]], [[1, 0], [1, 1]]):
    accuracy.add_batch(references=refs, predictions=preds)
res = accuracy.compute()
print(res)
"{'accuracy': 0.25}"


#3、 multi-evaluation indicator calculation
print("-----------#3、 multi-evaluation indicator calculation--------------")
clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])
# print(clf_metrics)

res = clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
print(res)
"""
{'accuracy': 0.6666666666666666,  2/3=0.67
'f1': 0.6666666666666666, 
'recall': 0.5, 'precision': 1.0}

计算过程如下：
TP = 1 # 第二个样本被正确预测为正例
FP = 0 # 没有样本被错误预测为正例
TN = 1 # 第一个和第三个样本被正确预测为反例
FN = 1 # 第三个样本被错误预测为反例
然后，我们可以使用上述的四个值来计算准确率（Accuracy）、召回率（Recall）、精确率（Precision）和F1值。
准确率（Accuracy）= （TP + TN）/ （TP + FP + TN + FN）=（1 + 1）/（1 + 0 + 1 + 1）= 2 / 3 ≈ 0.6666666666666666
召回率（Recall）= TP /（TP + FN）= 1 /（1 + 1）= 1 / 2 = 0.5
精确率（Precision）= TP /（TP + FP）= 1 /（1 + 0）= 1
F1值 = 2 * (精确率 * 召回率) / (精确率 + 召回率) = 2 * (1 * 0.5) / (1 + 0.5) = 1 / 1.5 ≈ 0.6666666666666666
因此，根据给定的predictions和references，我们得到以下结果：
准确率（Accuracy）≈ 0.6666666666666666
召回率（Recall）= 0.5
精确率（Precision）= 1
F1值 ≈ 0.6666666666666666
"""

# 4、result visualization
from evaluate.visualization import radar_plot
data = [
      {"accuracy": 0.99, "precision": 0.8, "f1": 0.95, "latency_in_seconds": 33.6},
      {"accuracy": 0.98, "precision": 0.87, "f1": 0.91, "latency_in_seconds": 11.2},
      {"accuracy": 0.98, "precision": 0.78, "f1": 0.88, "latency_in_seconds": 87.6}, 
      {"accuracy": 0.88, "precision": 0.78, "f1": 0.81, "latency_in_seconds": 101.6}
   ]
model_name = ['model_1', 'model_2', 'model_3', 'model_4']
plot = radar_plot(data=data, model_names=model_name)
import matplotlib.pyplot as plt
# plot.save("./evaluation_radart_plot.jpg")
plot.savefig("./evaluation_radart_plot.jpg")






















