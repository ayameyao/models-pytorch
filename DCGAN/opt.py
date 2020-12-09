
dataroot = "../data/celeba"
# dataroot = "../data/celeba"

# Number of workers for dataloader
# workers = 2
workers = 0

batch_size = 128

# 训练图像的空间大小,使用变压器将所有图像调整为该尺寸。
image_size = 64

# 训练的RGB图像通道
nc = 3

# z 的潜在矢量的大小（即生成器输入的大小）
nz = 100

# 生成器特征图大小
ngf = 64

# 判别器特征图大小
ndf = 64

# 迭代次数
num_epochs = 5

# 学习率
lr = 0.0002

# 用于Adam优化器的Beta1超参数
beta1 = 0.5

# 可用的GPU数量
ngpu = 1