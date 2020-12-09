import os
from torch.utils.data import random_split
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# define transformations for train
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define transformations for test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_training_dataloader(train_transform, batch_size=128, num_workers=2, shuffle=True):

    transform_train = train_transform
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_testing_dataloader(test_transform, batch_size=128, num_workers=0, shuffle=True):
    transform_test = test_transform
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

trainloader = get_training_dataloader(train_transform)
testloader = get_testing_dataloader(test_transform)

classes_dict = {0 : 'airplane', 1 : 'automobile', 2: 'bird', 3 : 'cat', 4 : 'deer', 5: 'dog', 6:'frog', 7 : 'horse', 8 : 'ship', 9 : 'truck'}


# plot 25 random images from training dataset
# fig, axs = plt.subplots(5, 5, figsize=(10,10))
    
# for batch_idx, (inputs, labels) in enumerate(trainloader):
#     for im in range(25):
#         image = inputs[im].permute(1, 2, 0)
#         i = im // 5
#         j = im % 5
#         axs[i,j].imshow(image.numpy()) #plot the data
#         axs[i,j].axis('off')
#         axs[i,j].set_title(classes_dict[int(labels[im].numpy())])
        
#     break;

# # set suptitle
# plt.suptitle('CIFAR-10 Images')
# plt.show()