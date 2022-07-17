import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        # 这里定义了残差块内连续的2个卷积层
        # nn.Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，
        # 同时以神经网络模块为元素的有序字典也可以作为传入参数。
        self.left = nn.Sequential(
            # 定义第一个卷积，默认卷积前后图像大小不变但可修改stride使其变化，通道可能改变
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),  # 数据的归一化处理
            nn.ReLU(inplace=True),
            # 定义第二个卷积，卷积前后图像大小不变，通道数不变
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        # 定义一条捷径，若两个卷积前后的图像尺寸有变化(stride不为1导致图像大小变化或通道数改变)，捷径通过1×1卷积用stride修改大小
        # 以及用expansion修改通道数，以便于捷径输出和两个卷积的输出尺寸匹配相加
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    # 定义前向传播函数，输入图像为x，输出图像为out
    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    # 定义初始函数，输入参数为残差块，默认参数为分类数10
    def __init__(self, ResBlock, num_classes=10):
        super().__init__()
        # 设置第一层的输入通道数
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            # 定义输入图片先进行一次卷积与批归一化，使图像大小不变，通道数由3变为64得两个操作
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 定义第一层，输入通道数64，有num_blocks[0]=2个残差块，残差块中第一个卷积步长自定义为1
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        # 定义第二层，输入通道数128，有num_blocks[1]=2个残差块，残差块中第一个卷积步长自定义为2
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # 定义全连接层，输入512个神经元，输出10个分类神经元
        self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    # 定义创造层的函数，在同一层中通道数相同，输入参数为残差块，通道数，残差块数量，步长
    def make_layer(self, block, channels, num_blocks, stride):
        # strides列表第一个元素stride表示第一个残差块第一个卷积步长，其余元素表示其他残差块第一个卷积步长为1
        strides = [stride] + [1] * (num_blocks - 1)
        # 创建一个空列表用于放置层
        layers = []
        # 遍历strides列表，对本层不同的残差块设置不同的stride
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))  # 创建残差块添加进本层
            self.inchannel = channels  # 更新本层下一个残差块的输入通道数或本层遍历结束后作为下一层的输入通道数
        return nn.Sequential(*layers)  # 返回层列表

    # 定义前向传播函数，输入图像为x，输出预测数据
    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # 经过一次4×4的平均池化
        out = out.view(out.size(0), -1)  # 将数据flatten平坦化
        out = self.fc(out)  # 全连接传播
        return out


def ResNet18():
    return ResNet(ResBlock)