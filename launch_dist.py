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