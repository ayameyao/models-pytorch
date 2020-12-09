import torch
import torch.nn as nn
from lstm import net, train_on_gpu, batch_size, train_loader


lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 4 
 
print_every = 100
clip=5 # gradient clipping
 
if(train_on_gpu):
    net.cuda()
 
net.train()
for e in range(epochs):
    # 初始化hidden state
    h = net.init_hidden(batch_size)
    counter = 0
 
    # batch loop
    for inputs, labels in train_loader:
        counter += 1
 
        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()
 
        # 为隐藏状态创建新变量，否则将在整个训练历史中反向传播
        h = tuple([each.data for each in h])
        # 零累积梯度
        net.zero_grad()
 
        # 模型输出
        output, h = net(inputs, h)
 
        # 计算损失并执行反向传播
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` 帮助防止RNN / LSTM中的爆炸梯度问题。
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
 
        # 损失
        if counter % print_every == 0:
            # 获取验证损失
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:
 
                # 为隐藏状态创建新变量，否则我们将在整个训练历史中反向传播
                val_h = tuple([each.data for each in val_h])
 
                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()
 
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
 
                val_losses.append(val_loss.item())
 
            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))