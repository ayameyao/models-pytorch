import numpy as np 
from collections import Counter
from data import words, reviews_split, labels

'''
embedding lookup要求输入的网络数据是整数。
最简单的方法就是创建数据字典：{单词：整数}。
然后将评论全部一一对应转换成整数，传入网络。
'''

# Encoding the words
# 建立将单词映射为整数的字典
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
 
# 使用字典来标记reviews_split中的每个评论
# 将标记化的评论存储在reviews_ints中
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])
 
# 词汇统计
print('Unique words: ', len((vocab_to_int)))  # 74072
 
# 在第一个review打印tokens
# print('Tokenized review: \n', reviews_ints[:1])



# Encoding the labels， 将标签 “positive” or "negative"转换为数值。
# 1=positive, 0=negative label conversion
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
 
# outlier review stats
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))


#消除长度为0的行
print('Number of reviews before removing outliers: ', len(reviews_ints))
 
## 从reviews_ints列表中删除长度为零的所有评论/标签。
 
# 获取长度为0的任何评论的索引
non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
 
# 删除长度为0的评论及其标签
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])
 
print('Number of reviews after removing outliers: ', len(reviews_ints))



# 固定句子长度为200
def pad_features(reviews_ints, seq_length):
    '''返回review_ints的功能，其中每个评论都用0填充或截断为输入seq_length.'''
    
    # 获取正确的行x 列形状
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
 
    # 对每个review,如下操作 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features
 
# 测试固定200长度的功能 
seq_length = 200
features = pad_features(reviews_ints, seq_length=seq_length)
 
## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
 
# print first 10 values of the first 30 batches 
print(features[:30,:10])

