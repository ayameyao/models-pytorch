import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
	# nhead – 多头注意力模型中的头数(default=8).
	def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
		super(TransformerModel, self).__init__()
		self.model_type = 'Transformer'
		self.pos_encoder = PositionalEncoding(ninp, dropout)
		encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.ninp = ninp
		self.decoder = nn.Linear(ninp, ntoken)

		self.init_weights()

	# 为该序列生成一个正方形mask，屏蔽的位置填充正无穷,未屏蔽的位置填充有0。
	# python中的正无穷或负无穷，使用float("inf")或float("-inf")来表示。
	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) #返回输入矩阵的转置（1-2维）
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	# src – 编码器的顺序（必需）.
	# src_mask – src序列的附加掩码（可选）.
	def forward(self, src, src_mask):
		src = self.encoder(src) * math.sqrt(self.ninp)
		src = self.pos_encoder(src)
		output = self.transformer_encoder(src, src_mask)
		output = self.decoder(output)
		return output


class PositionalEncoding(nn.Module):

	# d_model – 编码器/解码器输入中预期特征的数量 (default=512)
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model) # 返回一个全为标量 0 的张量

		# torch.arange返回一个1维张量， unsqueeze(1)返回一个新的张量，对输入的制定位置插入维度 1
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


# emsize = 200
# nhid = 200
# nlayers = 2
# nhead = 2
# dropout = 0.2
# print(TransformerModel(100, 2, nhead, nhid, nlayers))
# print(PositionalEncoding(512,0.2,5000))
		