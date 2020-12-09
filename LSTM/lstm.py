import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from train_test_split import train_x,train_y,val_x,val_y,test_x,test_y
from encoding import vocab_to_int
 
# 创建Tensor数据类型
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
 
batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# 获取训练集中的一个batch
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()
 
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


# Bi LSTM
train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


class SentimentRNN(nn.Module):
 
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True, drop_prob=0.5):

        super(SentimentRNN, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True,
                            bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
          self.fc = nn.Linear(hidden_dim, output_size)
          
        self.sig = nn.Sigmoid()
        
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
 
        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] #获取最后一个batch的表情
        
        # 输出最后一个sigmoid和hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        # 创建新的tensors大小为n_layers x batch_size x hidden_dim,
        # 并初始化为零，用于LSTM的隐藏状态和单元状态
        weight = next(self.parameters()).data
        
        number = 1
        if self.bidirectional:
           number = 2
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_()
                     )
        
        return hidden


# 是否使用双向LSTM
vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
bidirectional = False  #这里为True，为双向LSTM
 
net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)
 
print(net)