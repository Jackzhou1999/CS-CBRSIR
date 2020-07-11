from Model.Net import resnet18, resnet50
import torch
from torch.optim import Adam
from Process_data import Generator
from torch import nn
import numpy as np

def JOC_Loss(logits, Fi, Fc, criteon, y, alpha, lamta):
    label = torch.unique(y).cuda()
    L_soft = criteon(Fc, y)
    L_center = 0.
    L_Triplet = 0.
    if label.shape[0] != 8:
        return 0

    center = torch.zeros(size=(8, 512, 1, 1), requires_grad=True).cuda()
    for i in range(8):
        center[i] = torch.mean(logits[y == i], dim=0).cuda()

    for i in range(logits.shape[0]):


        L_center += torch.dist(logits[i], center[y[i]], p=2).pow(2)
        tmp = logits[torch.arange(logits.shape[0]) != i]
        tmpy = y[torch.arange(logits.shape[0]) != i]

        positive_anchor = tmp[tmpy == y[i].item()]
        a = np.random.randint(0, positive_anchor.shape[0], 1)
        positive_anchor = positive_anchor[a].reshape(512, 1, 1)

        negative_anchor = tmp[tmpy != y[i].item()]
        a = np.random.randint(0, negative_anchor.shape[0], 1)
        negative_anchor = negative_anchor[a].reshape(512, 1, 1)

        L_Triplet += (torch.dist(logits[i], positive_anchor, p=2) - torch.dist(logits[i], negative_anchor, p=2) + alpha)

    L_center /= 2.
    print("L_soft:", L_soft.item())
    print("L_center:", lamta*L_center.item())
    print("L_Triplet:", L_Triplet.item())


    return L_soft + L_Triplet + lamta*L_center



def train_f():
    lr = 0.001
    alpha = 0.3
    lamta = 0.0005
    epoch_n = 200
    train_batchsize = 128
    test_batchsize = 50
    train_iter = 75000 // train_batchsize
    test_iter = 5000 // test_batchsize

    imgsize = (256, 256)
    device = torch.device('cuda')
    F = resnet18().to(device)
    pretrained_dict = torch.load(r'/home/jackzhou/Desktop/Model_savealpha1.5/Fnet18alpha.pkl')['net']
    F.load_state_dict(pretrained_dict)
    for i, param in enumerate(F.parameters()):
        param.requires_grad = True

    optimizer = Adam(F.parameters(), lr=lr)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
    Gen = Generator(train_batchsize, imgsize)
    gen1 = Gen.generate(istrain=True, P=True, M=False)


    criteon1 = nn.CrossEntropyLoss(reduce=True, size_average=False).to(device)



    for epoch in range(epoch_n):
        print("EPOCH:", epoch)
        F.train()

        for batch_idx in range(train_iter):
            print("EPOCH:", epoch, "--", batch_idx, "/", train_iter)
            PAN, _, LABEL = next(gen1)

            if torch.cuda.is_available():
                PAN = torch.from_numpy(PAN).type(torch.FloatTensor)
                PAN = PAN.cuda()
                LABEL = torch.from_numpy(LABEL).type(torch.LongTensor)
                LABEL = LABEL.cuda()

            optimizer.zero_grad()
            logits, Fi, Fc = F(PAN)
            loss= JOC_Loss(logits, Fi, Fc, criteon1, LABEL, alpha, lamta)

            if loss == 0:
                print("fix!!!!")
                continue
            loss.backward()

            optimizer.step()

            print("LOSS:", loss.item())
            print()


        # if epoch%5 == 0:
        state = {'net': F.state_dict()}
        torch.save(state, r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/Fnet{}.pkl'.format(epoch))
        print("============================")


if __name__ == "__main__":
    np.random.seed(1)
    train_f()