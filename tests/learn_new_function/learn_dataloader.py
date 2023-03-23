import torch

# from torch.utils.datasets import SequentialSampler
# from torch.utils.datasets.sampler import (
#     BatchSampler,
#     RandomSampler,
#     Sampler,
#     SequentialSampler,
#     SubsetRandomSampler,
#     WeightedRandomSampler,
# )
"""
torch.utils.datasets.Dataset是代表这一数据的抽象类（也就是基类）。
我们可以通过继承和重写这个抽象类实现自己的数据类，只需要定义__len__和__getitem__这个两个函数。

DataLoader是Pytorch中用来处理模型输入数据的一个工具类。
组合了数据集（dataset） + 采样器(sampler)，并在数据集上提供单线程或多线程(num_workers )的可迭代对象。
在DataLoader中有多个参数，这些参数中重要的几个参数的含义说明如下：
    1. epoch：所有的训练样本输入到模型中称为一个epoch； 
    2. iteration：一批样本输入到模型中，成为一个Iteration;
    3. batchszie：批大小，决定一个epoch有多少个Iteration；
    4. 迭代次数（iteration）=样本总数（epoch）/批尺寸（batchszie）
    5. dataset (Dataset) – 决定数据从哪读取或者从何读取；
    6. batch_size (python:int, optional) – 批尺寸(每次训练样本个数,默认为１）
    7. shuffle (bool, optional) –每一个 epoch是否为乱序 (default: False)；
    8. num_workers (python:int, optional) – 是否多进程读取数据（默认为０);
    9. drop_last (bool, optional) – 当样本数不能被batchsize整除时，最后一批数据是否舍弃（default: False)
    10. pin_memory（bool, optional) - 如果为True会将数据放置到GPU上去（默认为false） 

"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集  数据放在了CIFAR10文件夹下

test_data = torchvision.datasets.CIFAR10(
    r"D:\Desktop\workdata\data/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)  # torch.Size([3, 32, 32]) 表示图像是RGB三通道，32*32尺寸的
print(target)  # 3  表示图像的类别是3

# batchsize=4  表示对数据读取的作用，表示一次性读取数据集中的4张图片，并且集合在一起进行返回
# 在定义test_loader时，设置了batch_size=4 表示一次性从数据集中读取四个数据
for data in test_loader:
    imgs, targets = data
    print(imgs.shape, targets)
    # output : torch.Size([4, 3, 32, 32]) tensor([3, 4, 5, 2])
    # 4表示batch_size=4，一次性取四张图片，后三个参数表示每张图片的信息，三通道RGB，尺寸为32*32
    # targets表示一次性取出的四张图片的target信息，分别是不同的labels 因为有0-9 10个不同的标签
    break

# 下面将epoch设置为2时，对数据集进行两次完整的遍历加载，同时设置batch_size=4
# 在定义test_loader时，设置了batch_size=4，表示一次性从数据集中取出4个数据
writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1
writer.close()


# 输入tensorboard --logdir="log文件所在路径"

# 关于 采样器 Sampler 和 数据加载器 DataLoader 目前需要着重学习Sampler
# sampler 负责决定读取数据时的先后顺序。Dataloader 负责装载数据并根据Sampler提供
# 所有的采样器都继承自Sampler这个类，主要是 __iter__ 指定每个step需要读取那些数据（产生迭代索引值）
class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    # 一个 迭代器 基类
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# SequentialSampler
# 其原理是首先在初始化的时候拿到数据集data_source，
# 之后在__iter__方法中首先得到一个和data_source一样长度的range可迭代器。每次只会返回一个索引值。
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    # 产生顺序 迭代器
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


"""
usage sample:
a = [1,5,78,9,68]
b = torch.utils.datasets.SequentialSampler(a)
for x in b:
    print(x)

>>> 0
    1
    2
    3
    4
"""


# RandomSampler
# 参数作用：data_source: 同上
#          num_samples: 指定采样的数量，默认是所有。
#          replacement: 若为True，则表示可以重复采样，即同一个样本可以重复采样，这样可能导致有的样本采样不到。
#                       所以此时我们可以设置num_samples来增加采样数量使得每个样本都可能被采样到。
class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


# SubsetRandomSampler
class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


"""
这个采样器常见的使用场景是将训练集划分成训练集和验证集，示例如下：
n_train = len(train_dataset)
split = n_train // 3
indices = random.shuffle(list(range(n_train)))
train_sampler = torch.utils.datasets.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.datasets.sampler.SubsetRandomSampler(indices[:split])
train_loader = DataLoader(..., sampler=train_sampler, ...)
valid_loader = DataLoader(..., sampler=valid_sampler, ...)
"""


# BatchSampler   默认的读取数据的格式：使用的都是batch sampler。
class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # 批次采样
    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.datasets.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# ========================================
class DataLoader(object):
    ...

    def __next__(self):
        if self.num_workers == 0:
            indices = next(self.sample_iter)
            batch = self.collate_fn([self.dataset[i] for i in indices])  # this line
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch


# collate_fn的作用就是将一个batch的数据进行合并操作。默认的collate_fn是将img和label分别合并成imgs和labels，
# 所以如果你的__getitem__方法只是返回 img, label,那么你可以使用默认的collate_fn方法，
# 但是如果你每次读取的数据有img, box, label等等，那么你就需要自定义collate_fn来将对应的数据合并成一个batch数据，这样方便后续的训练步骤。

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

# 使用dataloader时加入collate_fn参数，即可合并样本列表以形成小批量的Tensor对象，
# 如果你的标签不止一个的话，还可以支持自定义，在上述方法中再额外添加对应的label即可。