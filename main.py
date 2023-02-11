import torch
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from ddpm import DDPMTrainer, DDPMSampler
from unet import UNet

epoch = 100
lr = 0.0002
batch_size = 128

T = 1000
beta_1, beta_T = 0.0001, 0.02

def train():
    cifar10 = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(cifar10, batch_size, shuffle=True, num_workers=4, drop_last=True)

    net = UNet(T=T, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    trainer = DDPMTrainer(T, beta_1, beta_T, net)
    for i in range(epoch):
        for x_0, _ in dataloader:
            optim.zero_grad()
            loss = trainer(x_0).mean()
            loss.backward()
            optim.step()
            print('loss: %.4f' % loss)

        

if __name__ == '__main__':
    train()