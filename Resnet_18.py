from resnet import ResNet18
# Use the ResNet18 on Cifar-10
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import argparse
import os

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
args = parser.parse_args()

# set hyperparameter
EPOCH = 20
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01

# prepare dataset and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# labels in CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define ResNet18 模型
net = ResNet18().to(device)

# define loss funtion & optimizer # 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# train
if __name__ == '__main__':
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(pre_epoch, EPOCH):  # 循环训练回合，每回合会以批量为单位训练完整个训练集，一共训练EPOCH个回合
                print('\nEpoch: %d' % (epoch + 1))
                # 每一训练回合初始化累计训练损失函数为0.0，累计训练正确样本数为0.0，训练样本总数为0//start为开始计时的时间点
                net.train()
                sum_loss = 0.0
                correct = 0.0
                correct_3 = 0.0
                total = 0.0

                for i, data in enumerate(trainloader, 0):  # 循环每次取一批量的图像与标签
                    # prepare dataset
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)  # 将图像和标签分别搬移至指定设备上
                    optimizer.zero_grad()  # 优化器梯度清零

                    # forward & backward
                    outputs = net(inputs)  # 将批量图像数据inputs输入网络模型net，得到输出批量预测数据ouputs
                    loss = criterion(outputs, labels)  # 计算批量预测标签outputs与批量真实标签labels之间的损失函数loss
                    loss.backward()  # 对批量损失函数loss进行反向传播计算梯度
                    optimizer.step()  # 优化器的梯度进行更新，训练所得参数也更新

                    # print ac & loss in each batch
                    sum_loss += loss.item()  # 将本批量损失函数loss加至训练损失函数累计sum_loss中
                    # top_1
                    # 返回输出的最大值（不care）和最大值对应的索引，dim=1表示输出所在行 的最大值，
                    # 10分类问题对应1行由10个和为1的概率构成，返回概率最大的作为预测索引
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()  # 将本批量预测正确的样本数加至累计预测正确样本数correct中
                    # top_3
                    maxk = max((1, 3))
                    y_resize = labels.view(-1, 1)
                    _, pred = outputs.topk(maxk, 1, True, True)
                    correct_3 += torch.eq(pred, y_resize).sum().float().item()
                    total += labels.size(0)  # 将本批量训练的样本数，加至训练样本总数
                    Train_epoch_loss = sum_loss / (i + 1)
                    Train_epoch_acc = 100. * correct / total
                    Train_epoch_acc_3 = 100. * correct_3 / total
                    print('epoch:%d | iter:%d | Loss: %.03f | top1Acc: %.3f%% | top3Acc: %.3f%%'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                             100. * correct_3 / total))
                    f2.write('epoch:%03d | iter:%05d | Loss: %.03f | top1Acc: %.3f%% | top3Acc: %.3f%%'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total,
                                100. * correct_3 / total))
                    f2.write('\n')
                    f2.flush()

                # get the ac with testdataset in each epoch # 每训练完一个epoch测试一下准确率
                print('Waiting Test...')
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        """
                        loss = criterion(outputs, labels)  # 计算批量预测标签outputs与批量真实标签labels之间的损失函数loss
                        # loss.requires_grad_(True)  # 传入一个参数requires_grad=True, 这个参数表示是否对这个变量求梯度， 默认的是False, 也就是不对这个变量求梯度。
                        loss.backward()  # 对批量损失函数loss进行反向传播计算梯度
                        optimizer.step()  # 优化器的梯度进行更新，训练所得参数也更新
                        # print ac & loss in each batch
                        sum_loss += loss.item()  # 将本批量损失函数loss加至训练损失函数累计sum_loss中
                        """
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('test acc: %.3f%%' % (100 * correct / total))
                    top1Acc = 100. * correct / total

                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, top1Acc))
                    f.write('\n')
                    f.flush()

                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if top1Acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, top1Acc))
                        f3.close()
                        best_acc = top1Acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
