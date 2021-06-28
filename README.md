# Pytorch分布式训练

## 结构介绍

Pytorch 1.x 的多机多卡计算模型并没有采用主流的 Parameter Server 结构，而是直接用了 Uber 的 Horovod 的形式，也是百度开源的 RingAllReduce 算法。

采用 PS 计算模型的分布式（如MXNet默认的分布式方法），通常会遇到网络的问题，随着 worker 数量的增加，其加速比会迅速的恶化，需要借助其他辅助技术。

而 Uber 的 Horovod，采用的 RingAllReduce 的计算方案，其特点是网络单次通信量不随着 worker(GPU) 的增加而增加，是一个恒定值。

在 RingAllReduce 中，GPU 集群被组织成一个逻辑环，每个 GPU 只从左邻居接受数据、并发送数据给右邻居，即每次同步每个 GPU 只获得部分梯度更新，等一个完整的 Ring 完成，每个 GPU 都获得了完整的参数。通讯成本是系统中 GPU 之间最慢的连接决定的。

原理图如下：


![bj-4a8f4ccb6d7c1099abbd168a8dbba1d12370ea8a](https://user-images.githubusercontent.com/35672492/123613396-f75f0d00-d835-11eb-9ad6-fddb6c34aa7c.png)
![bj-641e6112b5bb1f332146f2e2763f9c2ce6b0356e](https://user-images.githubusercontent.com/35672492/123613407-fa59fd80-d835-11eb-9e09-72a04437fa47.png)
![bj-b7eb6ee97dd909a9df0e56feb17b4fdeef0eed38](https://user-images.githubusercontent.com/35672492/123613424-fd54ee00-d835-11eb-89c9-4accee01a1b1.png)

可见，五次迭代后，所有的GPU都更新了值。

Pytorch 中通过 `torch.distributed` 包提供分布式支持，包括 GPU 和 CPU 的分布式训练支持。Pytorch 分布式目前只支持 Linux。

## 基本概念
以下是Pytorch分布式训练中常见的一些概念：

 - `group`: 进程组。默认情况下，只有一个进程组，一个 `job `即为一个组，也即一个 `world`。
当需要进行更加精细的通信时，可以通过 `new_group` 接口，使用 `word` 的子集，创建新组，用于集体通信等。

 - `world_size`: 表示全局进程个数。

 - `rank`: 表示进程序号，用于进程间通讯，表征进程优先级。`rank = 0` 的主机为 `master` 节点。

 - `local_rank`: 进程内，GPU 编号，非显式参数，由 `torch.distributed.launch` 内部指定。比方说， `rank = 3`，`local_rank = 0` 表示第 3 个进程内的第 1 块 GPU。

## 使用流程

 1. 在使用 `distributed` 包的任何其他函数之前，需要使用 `init_process_group` 初始化进程组，同时初始化 `distributed` 包。
 2. 如果需要进行小组内集体通信，用 `new_group` 创建子分组。***非必须项***
 3. 创建分布式并行模型 `DDP(model, device_ids=device_ids)`
 4. 为数据集创建 `Sampler`
 5. 使用启动工具 `torch.distributed.launch` 在每个主机上执行一次脚本，开始训练。也可在代码中使用 `torch.multiprocessing.spawn`并在每个主机上执行一次脚本，开始训练。
 6. 使用 `destory_process_group()` 销毁进程组。***非必须项***
 
## 案例演示
### 一、使用 torch.multiprocessing 启动分布式

#### ①. 代码修改

1.载入需要使用的package，其中`torch.multiprocessing`和`torch.distributed`为分布式所需要使用的包：
```python
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
```

2.写一个神经网络模型：
```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=10)
        
    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.maxpool3(self.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4)
        x = self.fc1(x)
        return x
```

3.定义`main`函数：
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    #########################################################
    args.world_size = args.gpus * args.nodes                
    os.environ['MASTER_ADDR'] = '172.16.16.5'               
    os.environ['MASTER_PORT'] = '8888'                      
    mp.spawn(train, nprocs=args.gpus, args=(args,))         
    #########################################################
```
其中，`args.nodes`定义了分布式训练任务中机器的数量，`args.gpus`定义了每台机器上 GPU 的数量，`args.nr`为机器在分布式任务中的`rank`号，从0起算。`args.world_size`为任务中进程数，一个 GPU 一个进程时（强烈推荐此方式），则为 GPU 总数。

`MASTER_ADDR`系统变量为主机的 IP 地址，`MASTER_PORT`为主机的端口号，主机默认为`rank`号为0的机器。

例如：我们使用两台八卡机器做分布式训练，那么`args.nodes`应为`2`，`args.gpus`应为`8`，`args.world_size`则是`16`，作为主机的机器的`args.nr`应为`0`，另一台则是`1`。

然后，使用 `torch.multiprocessing` 分配任务，其中`train`为训练函数（见下一步），`nprocs`为分配的进程数，需要与本机 GPU 数一致。`args`为传入`train`函数的实参。本函数详情可见[官方文档](https://pytorch.org/docs/stable/multiprocessing.html)。



4.接下来，定义训练函数`train`，其中需要添加部分代码以实现分布式训练：

```python
def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
```
其中，`rank`函数代表了当前运行脚本的 GPU 在整个分布式任务中的编号，`init_process_group`用来初始化进程组。

`backend`指定当前进程要使用的通信后端，支持的通信后端有 `gloo`，`mpi`，`nccl `。建议用 `nccl`。
`init_method `指定当前进程组初始化方式，默认使用`env://`配合第三步中系统参数，也可以修改为`tcp://主机IP：端口号`的形式。
`world_size`告知整个任务的进程数，也就是 GPU 数目。
`rank`为当前进程的编号。`rank=0`为主进程，即`master`节点。


```python
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
```

`torch.manual_seed(0)`用来统一每个进程中随机部分，使得模型可以统一。

```python
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=args.world_size,
        rank=rank
    )
    # shuffle use False, add sampler=train_sampler
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
```
`DistributedDataParallel` 将给定的模型进行分布式封装，模型会被复制到每台机器的每个 GPU 上，每一个模型的副本处理输入的一部分。

`DistributedSampler`用于切分数据集。在读取数据时，需要将`shuffle`关闭，取而代之的是`sampler`。

`pin_memory=True`，则意味着生成的 Tensor 数据最开始是属于内存中的锁页内存，这样将内存的 Tensor 转义到 GPU 的显存就会更快一些。建议计算机内存充足时开启。

主要训练部分：
```python
    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Overlapping transfer if pinned memory
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
```
只打印每台机器 0 号 GPU 的日志，也可修改成只打印 master 节点的日志，即`global_rank = 0`的日志。


5.最后：
```python
if __name__ == '__main__':
    main()
```

#### ②. 运行脚本
以两台机器，每台一卡为例，两台机器的控制台分别运行：
```
python mnist_dist_py.py -n 2 -g 1 -nr 0 --epochs 5
python mnist_dist_py.py -n 2 -g 1 -nr 1 --epochs 5
```
如果需要查看NCCL通讯日志，则可输入
```terminal
export NCCL_DEBUG=INFO
```
以下是 master 节点所在机器上的日志效果图：
![image](https://user-images.githubusercontent.com/35672492/123612967-9afbed80-d835-11eb-9fe0-d0e55002b1ec.png)


### 二、使用 torch.distributed.launch 启动分布式

#### ①. 代码修改
基于上述方法修改，我们将训练函数写在`if __name__ == '__main__':`后，添加`rank`和`local_rank`参数读取。删除整个`main`函数，不要保留`args`参数。

对应的将代码中原本的`gpu`参数用`local_rank`替代，原本的`args.world_size`用`dist.get_world_size()`获取。

`init_process_group`中参数只保留`backend`。以下为修改后的代码：
```python
import os
from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.relu = nn.ReLU()
        #Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        #Max pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        #Max pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        #Convolution layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        #Max pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        # fully connected layer (softmax)
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.maxpool2(self.relu(self.conv2(x)))
        x = self.maxpool3(self.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(backend='nccl')
    torch.manual_seed(0)
    model = ConvNet()

    device = torch.device("cuda", local_rank)

    model = model.to(device)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        num_replicas=dist.get_world_size(),
        rank=local_rank
    )
    # shuffle use False, add sampler=train_sampler
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,
                                               pin_memory=True, sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    print('Total step: ', total_step)
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            # Overlapping transfer if pinned memory
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and local_rank == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    10,
                    i + 1,
                    total_step,
                    loss.item())
                )
    if local_rank == 0:
        print("Training complete in: " + str(datetime.now() - start))
```
#### ②. 运行脚本
以两台机器，每台一卡为例，两台机器的控制台分别运行：
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="172.16.16.5" --master_port=22222 mnist_dist.py

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="172.16.16.5" --master_port=22222 mnist_dist.py
```
效果与`torch.multiprocessing`方法一致。
