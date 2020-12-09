import torch
import torch.nn as nn
import numpy as np 
from lstm import net, train_on_gpu, batch_size, test_loader
from train import criterion
 

test_on_gpu=torch.cuda.is_available()
if(test_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

if(test_on_gpu):
    net.cuda()

test_losses = []
num_correct = 0
 
# 初始化hidden state
h = net.init_hidden(batch_size)


net.eval()
# 遍历testdata
for inputs, labels in test_loader:

    h = tuple([each.data for each in h])
 
    if(train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()
    
    # predicted outputs
    output, h = net(inputs, h)
    
    # loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    
    # 将输出概率转换为预测的类别（0或1）
    pred = torch.round(output.squeeze()) 
    
    # 将预测与真实标签进行比较
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))
 
# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))