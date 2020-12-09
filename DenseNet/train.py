import os
import torch
import torch.nn as nn
from torch.optim import SGD,Adam,lr_scheduler
import pandas as pd
import time
import numpy as np
import data
from densenet import densenet121


epochs = 10
learning_rate = 0.001

print("---------------------- start to training --------------------------\n")
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print("lgy's machine is: ", device)

print("---------------------- densenet121 --------------------------\n")
model = densenet121()


criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

#打印模型的参数
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor,"\t",model.state_dict()[param_tensor].size())
#打印优化器的state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])

train_stats = pd.DataFrame(
	columns = ['Epoch', 'Time per epoch', 'Avg time per step', 
	'Train loss', 'Train accuracy', 'Train top-3 accuracy',
	'Test loss', 'Test accuracy', 'Test top-3 accuracy']) 

print("---------------------- device --------------------------\n")
model.to(device)

steps = 0
running_loss = 0
for epoch in range(epochs):
    
    since = time.time()
    
    train_accuracy = 0
    top3_train_accuracy = 0 
    for inputs, labels in data.trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 计算 top-1 精确度
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # 计算 top-3 精确度
        np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
        target_numpy = labels.cpu().numpy()
        top3_train_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])
        
    time_elapsed = time.time() - since
    
    test_loss = 0
    test_accuracy = 0
    top3_test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in data.testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # top-1 accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # top-3 accuracy
            np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_test_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

    len_train = len(data.trainloader)
    len_test = len(data.testloader)
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Time per epoch: {time_elapsed:.4f}.. "
          f"Average time per step: {time_elapsed/len_train:.4f}.. "
          f"Train loss: {running_loss/len_train:.4f}.. "
          f"Train accuracy: {train_accuracy/len_train:.4f}.. "
          f"Top-3 train accuracy: {top3_train_accuracy/len_train:.4f}.. "
          f"Test loss: {test_loss/len_test:.4f}.. "
          f"Test accuracy: {test_accuracy/len_test:.4f}.. "
          f"Top-3 test accuracy: {top3_test_accuracy/len_test:.4f}")

    train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':time_elapsed, 'Avg time per step': time_elapsed/len(data.trainloader),
     'Train loss' : running_loss/len_train, 
     'Train accuracy': train_accuracy/len_train, 
     'Train top-3 accuracy':top3_train_accuracy/len_train,
     'Test loss' : test_loss/len(data.testloader), 
     'Test accuracy': test_accuracy/len(data.testloader), 
     'Test top-3 accuracy':top3_test_accuracy/len(data.testloader)}, 
     ignore_index=True)

    running_loss = 0
    model.train()

train_stats.to_csv('train_log_DenseNet121.csv')
torch.save(model.state_dict(), 'pth/densenet121.pth')