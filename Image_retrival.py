from Process_data import Generator
from Model.Net import resnet18
import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
# np.random.seed(1)

def retrival():
    gen = Generator(batch_size=10, image_size=(256, 256)).generate(istrain=False, P=True, M=True)

    device = torch.device('cuda')
    ALL_TEST_PAN = list()
    ALL_TEST_MUL = list()
    ALL_TEST_Label = list()

    flag1 = True
    if flag1:
        F = resnet18().to(device)

        F_dict = torch.load(r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/Fnet16.pkl')['net']

        # F_dict = torch.load(r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/Fnet18alpha.pkl')['net']
        F.load_state_dict(F_dict)
        for i, param in enumerate(F.parameters()):
            param.requires_grad = False

    flag2 = False
    if flag2:
        G = resnet18().to(device)
        G_dict = torch.load(r'/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/Gnet.pkl')['net']
        G.load_state_dict(G_dict)
        for i, param in enumerate(G.parameters()):
            param.requires_grad = False



    for i in range(500):
        PAN, MUL, LABEL = next(gen)
        if torch.cuda.is_available():
            PAN = torch.from_numpy(PAN).type(torch.FloatTensor)
            PAN = PAN.cuda()
            MUL = torch.from_numpy(MUL).type(torch.FloatTensor)
            MUL = MUL.cuda()

        if flag1:
            logits, Fi, Fc = F(PAN)
            ALL_TEST_PAN.append(logits.cpu().numpy())
        if flag2:
            logits1, Fi1, Fc1 = G(MUL)
            ALL_TEST_MUL.append(logits1.cpu().numpy())
        ALL_TEST_Label.append(LABEL)

    ALL_TEST_PAN = np.concatenate(ALL_TEST_PAN, axis=0)
    ALL_TEST_Label = np.concatenate(ALL_TEST_Label)
    accu = 0.
    for i in range(1000):
        idx = np.random.randint(0, 5000, 1)
        image = ALL_TEST_PAN[idx].reshape(1, -1)
        label = ALL_TEST_Label[idx][0]
        ALL_TEST_PAN1 = ALL_TEST_PAN[np.arange(0, 5000) != idx]
        ALL_TEST_Label1 = ALL_TEST_Label[np.arange(0, 5000) != idx]
        # np.random.seed(1)
        distance = cdist(image, ALL_TEST_PAN1.reshape(4999, 512), metric='cosine')
        num = 100
        idx1 = np.argsort(distance[0])[:num]
        print("==========================================================================")
        print("检索图片LABEL={}, 共检索{}张同类图片,准确率：{}".format(label, num, np.sum(ALL_TEST_Label1[idx1].reshape(1, -1) == label)/num))
        # print(ALL_TEST_Label1[idx1])
        accu += np.sum(ALL_TEST_Label1[idx1].reshape(1, -1) == label)/num
        print("==========================================================================")

    print(accu/1000)
    X_embedded = TSNE(n_components=2).fit_transform(ALL_TEST_PAN.reshape(-1, 512))
    print(X_embedded.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=3, c=ALL_TEST_Label)
    plt.show()


if __name__ == "__main__":
    retrival()