# import argparse
import os
# import random
import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from data import device, dataloader
from opt import num_epochs, nz, lr, beta1, ngpu
from main import netD, netG


# 损失函数和优化器

# 初始化 BCELoss function
criterion = nn.BCELoss()

# 创建可视化生成器的进程
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 将实际标签定义为 1，将假标签定义为 0
real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# 训练

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        # (1) 更新 D network: maximize log(D(x)) + log(1 - D(G(z)))
        # 训练所有的 real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        # label = torch.full((b_size,), real_label, device=device)
        label = torch.full((b_size,), real_label, device=device, dtype=torch.float)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
       
        # 计算所有的 real batch 损失
        errD_real = criterion(output, label)
        
        # 计算 D 反向传播的梯度
        errD_real.backward()
        D_x = output.mean().item()

        ## 训练所有的 fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 通过 G 批量生成图片
        fake = netG(noise)
        label.fill_(fake_label)
        # 用 D 对假标签分类
        output = netD(fake.detach()).view(-1)
        # 计算 D 在所有 fake batch上的损失
        errD_fake = criterion(output, label)
        # 计算梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 对所有的 real 和 fake batches 梯度求和
        errD = errD_real + errD_fake
        # 更新 D
        optimizerD.step()


        # (2) 更新 G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake).view(-1)
        # 计算当前输出 G 的损失
        errG = criterion(output, label)
        # 计算 G 的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # 输出训练统计
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 保存损失值用于后面画图
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 通过将G的输出保存在fixed_noise中，追踪生成器运行情况
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1