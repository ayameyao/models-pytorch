import numpy as np 
from string import punctuation
from collections import Counter

#读取数据
with open('./data/reviews.txt', 'r') as f:
	reviews = f.read()
with open('./data/labels.txt', 'r') as f:
	labels = f.read()

# print(reviews[:1000])
# print(labels[:1000])

#预处理
#去除标点符号
reviews = reviews.lower()  #小写-标准化
all_text = ''.join([c for c in reviews if c not in punctuation])

#按照新行和空格分开
reviews_split = all_text.split('\n')
# all_text = ''.join(reviews_split)
all_text = ''.join(reviews_split)

#创建新的单词列表
words = all_text.split()
print(words[:30])








