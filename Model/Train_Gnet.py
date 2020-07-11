from Model.Net import resnet18, resnet50
import torch
from torch.optim import Adam
from Process_data import Generator
from torch import nn
import numpy as np


def train_g():
    lr = 0.001
    alpha = 0.3
    lamta = 0.0005
    epoch_n = 200
    train_batchsize = 96
    test_batchsize = 50
    train_iter = 75000 // train_batchsize
    test_iter = 5000 // test_batchsize

    imgsize = (256, 256)
    device = torch.device('cuda')
    G = resnet18().to(device)
    F = resnet18().to(device)

    pretrained_dict = torch.load(r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/test.pkl')['net']
    G.load_state_dict(pretrained_dict)
    F.load_state_dict(pretrained_dict)

    for i, param in enumerate(F.parameters()):
        param.requires_grad = False

    for i, param in enumerate(G.parameters()):
        # print(i, param.shape)
        if i >= 15:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = Adam(filter(lambda p: p.requires_grad, G.parameters()), lr=lr)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
    gen1 = Generator(train_batchsize, imgsize).generate(istrain=True, P=True, M=True)

    criteon = torch.nn.MSELoss().to(device)

    for epoch in range(epoch_n):
        F.train()
        for batch_idx in range(train_iter):
            PAN, MUL, _ = next(gen1)

            if torch.cuda.is_available():
                PAN = torch.from_numpy(PAN).type(torch.FloatTensor)
                PAN = PAN.cuda()
                MUL = torch.from_numpy(MUL).type(torch.FloatTensor)
                MUL = MUL.cuda()

            optimizer.zero_grad()
            logits, Fi, Fc = F(PAN)
            logits1, Fi1, Fc1 = G(MUL)
            loss = criteon(logits, logits1)
            loss.backward()
            optimizer.step()

            print(loss.item())
            state = {'net': G.state_dict()}
            torch.save(state, r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/Gnet.pkl')
        print("============================")


if __name__ == "__main__":
    train_g()