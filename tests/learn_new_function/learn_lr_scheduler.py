# torch.optim.lr_scheduler 模块提供了一些根据 epoch 迭代次数来调整学习率 lr 的方法。
# 为了能够让损失函数最终达到收敛的效果，通常 lr 随着迭代次数的增加而减小时能够得到较好的效果。
# torch.optim.lr_scheduler.ReduceLROnPlateau 则提供了基于训练中某些测量值使学习率动态下降的方法。

#TODO demo
"""
model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        # 1.进行参数的更新
        optimizer.step()
    # 2.对学习率进行更新    
    scheduler.step()
# 注意：现在，在很多祖传代码中，scheduler.step()的位置可能是在参数更新optimizer.step()之前
# 检查您的pytorch版本如果是V1.1.0+，那么需要将scheduler.step()在optimizer.step()之后调用	    

"""


# 优化器 Optimizer 以Adam为例 （所有 optimizers 都继承自 torch.optim.Optimizer 类）

# 对于 class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
"""
params (iterable)：需要优化的网络参数，传进来的网络参数必须是Iterable。
    优化一个网络，网络的每一层看做一个parameter group，一整个网络就是parameter groups
    （一般给赋值为net.parameters()——generator的字典）；

    优化多个网络，有两种方法：
        多个网络的参数合并到一起，形如[*net_1.parameters(), *net_2.parameters()]
        或itertools.chain(net_1.parameters(), net_2.parameters())；

        当成多个网络优化，让多个网络的学习率各不相同，形如[{‘params’: net_1.parameters()}, {‘params’: net_2.parameters()}]

lr (float, optional)：学习率；
betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))；
eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)；
weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)；
amsgrad (boolean, optional) – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)。

"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import itertools

initial_lr = 0.1

class model(nn.Module):
    # 简单定义一个网络类，并没有实现网络应有的功能，只是用来定义optimizer的。
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

# 实例化网络
net_1 = model()
net_2 = model()

# 实例化一个Adam对象
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)

# print("******************optimizer_1*********************")
# print("optimizer_1.defaults：", optimizer_1.defaults)
# print("optimizer_1.param_groups长度：", len(optimizer_1.param_groups))
# print("optimizer_1.param_groups一个元素包含的键：", optimizer_1.param_groups[0].keys())
# print('==========================')


# optimizer_2 = torch.optim.Adam([*net_1.parameters(), *net_2.parameters()], lr = initial_lr)
# # optimizer_2 = torch.opotim.Adam(itertools.chain(net_1.parameters(), net_2.parameters())) # 和上一行作用相同
# print("******************optimizer_2*********************")
# print("optimizer_2.defaults：", optimizer_2.defaults)
# print("optimizer_2.param_groups长度：", len(optimizer_2.param_groups))
# print("optimizer_2.param_groups一个元素包含的键：", optimizer_2.param_groups[0].keys())
# print('==========================')

# optimizer_3 = torch.optim.Adam([{"params": net_1.parameters()}, {"params": net_2.parameters()}], lr = initial_lr)
# print("******************optimizer_3*********************")
# print("optimizer_3.defaults：", optimizer_3.defaults)
# print("optimizer_3.param_groups长度：", len(optimizer_3.param_groups))
# print("optimizer_3.param_groups一个元素包含的键：", optimizer_3.param_groups[0].keys())

"""
******************optimizer_1*********************
optimizer_1.defaults： {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}
optimizer_1.param_groups长度： 1
optimizer_1.param_groups一个元素包含的键： dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable'])
==========================
******************optimizer_2*********************
optimizer_2.defaults： {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}
optimizer_2.param_groups长度： 1
optimizer_2.param_groups一个元素包含的键： dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable'])
==========================
******************optimizer_3*********************
optimizer_3.defaults： {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False}
optimizer_3.param_groups长度： 2
optimizer_3.param_groups一个元素包含的键： dict_keys(['params', 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable'])
"""

from torch.optim.lr_scheduler import LambdaLR
# 实例化一个LambdaLR对象。lr_lambda是根据epoch更新lr的函数。
scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch: 1/(epoch+1))

print("初始化的学习率：", optimizer_1.defaults['lr'])

# 模仿训练的epoch
for epoch in range(1, 11):
    # train
    # 更新网络参数（这里省略了loss.backward()）。
    optimizer_1.zero_grad()
    optimizer_1.step()
    print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
    scheduler_1.step()
