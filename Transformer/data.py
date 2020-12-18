import io
import torch
from torchtext.utils import download_from_url, extract_archive
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import TransformerModel


# url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
# test_filepath, valid_filepath, train_filepath = extract_archive(download_from_url(url))

test_filepath = "./data/wikitext-2/wiki.test.tokens"
valid_filepath = "./data/wikitext-2/wiki.valid.tokens"
train_filepath =  "./data/wikitext-2/wiki.train.tokens"

# get_tokenizer函数的作用是创建一个分词器，将语料喂给相应的分词器，可以根据不同分词函数的规则完成分词，
# 分词器支持’basic_english’，‘spacy’，‘moses’，‘toktok’，‘revtok’，'subword’等规则。
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding='utf-8'))))


def data_process(raw_text_iter):
	data = [torch.tensor([vocab[token] for token in tokenizer(item)],
		dtype = torch.long) for item in raw_text_iter]
	# numel()函数：返回数组中元素的个数
	return torch.cat(tuple(filter(lambda t: t.numel()>0, data)))

train_data = data_process(iter(io.open(train_filepath, encoding='utf-8')))
val_data = data_process(iter(io.open(valid_filepath, encoding='utf-8')))
test_data = data_process(iter(io.open(test_filepath, encoding='utf-8')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
	# 将数据集分割成`bsz`批次
	nbatch = data.size(0) // bsz
	# 将不能整除（剩余）的多余数据裁减掉
	data = data.narrow(0, 0, nbatch * bsz)
	# 将数据平均分配到`bsz`个批次
	data = data.view(bsz, -1).t().contiguous()
	return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, batch_size)
test_data = batchify(test_data, batch_size)


# get_batch()为transformer模型生成输入和目标序列
bptt = 35
def get_batch(source, i):
	seq_len = min(bptt, len(source) - 1 - i)
	data = source[i:i+seq_len]
	target = source[i+1:i+1+seq_len].reshape(-1)
	return data, target

#Initiate an instance
ntokens = len(vocab.stoi) # 词汇表的大小
emsize = 200 # 嵌入层维度
nhid = 200 # nn.TransformerEncoder 中的前馈网络模型的维度
nlayers = 2 # nn.TransformerEncoder中nn.TransformerEncoderLayer的层数
nhead = 2 # 多头注意力（multiheadattention）模型头的数量
dropout = 0.2 # dropout 的概率
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)



