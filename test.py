from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
from Ex3_4 import OneHotEncoding
datasets = load_dataset("clapAI/MultiLingualSentiment")
datasets = datasets.filter(lambda x: x["language"]=="en" )
datasets = datasets.remove_columns(["source","domain","language"])
labels=[]
for label in datasets["train"]["label"]:
    labels.append(label)
counts = Counter(labels)
print(counts) # Counter({'negative': 442928, 'positive': 412535, 'neutral': 360246})
string = " ".join(labels)
labels = OneHotEncoding(string).one_hot()
print(labels)
# print(string)
print(counts)
