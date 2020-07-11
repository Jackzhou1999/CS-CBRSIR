import torch
#
# a = torch.randint(high=10, size=(10,))
#
# print(a)
# print(a == 1)

import scipy.io
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import random

# ============================================
# if istrain:
#     train_idx = idx[:self.train_size]
#     train = []
#     for i in range(self.train_iter):
#         id = np.sort(train_idx[i * self.batch_size:(i + 1) * self.batch_size])
#         train.append(id)
#
#     while True:
#         for i in range(self.train_iter):
#             id = train[i]
#             PAN = np.repeat(self.PAN_IMAGES[id], 4, axis=1)
#             yield PAN
# else:
#     test_idx = idx[self.train_size:]
#     test = []
#     for i in range(self.test_iter):
#         id = np.sort(test_idx[i * self.batch_size:(i + 1) * self.batch_size])
#         test.append(id)
#
#     for i in range(self.test_size):
#         pass
# ===================================================================

def load_dataset():
    with h5py.File('Dataset/DSRSID.mat', 'r') as f:

        # PAN_IMAGES = np.array(f["PAN_IMAGES"])[:1000]
        # PAN_IMAGES = np.repeat(PAN_IMAGES, 4, axis=1)
        # print(PAN_IMAGES.shape, type(PAN_IMAGES), PAN_IMAGES.dtype)
        save = []

        for i in range(80000):
            # print(f["MUL_IMAGES"][0])
            MUL_IMAGES = np.array(f["MUL_IMAGES"][:10])
            # print(MUL_IMAGES[0].shape)
            tmp = MUL_IMAGES[0]
            # print(tmp)
            tmp = np.transpose(tmp, axes=(1, 2, 0))
            tmp = Image.fromarray(tmp)
            tmp = tmp.resize(size=(256, 256), resample=Image.BICUBIC)
            tmp = np.expand_dims(np.array(tmp, dtype=np.float64).transpose((2, 0, 1)), axis=0)
            save.append(tmp)
            # print(tmp.shape)
            # print(tmp)
            if (i+1)%128 == 0:
                a = np.concatenate(save, axis=0)
                print(a.shape)
                save.clear()

        # print(MUL_IMAGES.shape)
        # LAND_COVER_TYPES = f["LAND_COVER_TYPES"]
        # print(type(PAN_IMAGES), PAN_IMAGES.shape)


if __name__ == "__main__":
    # load_dataset()
    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[2, 3], [5, 6]])
    # x = torch.randperm(4)
    # y = torch.randperm(4)
    # x = x.float()
    # y = y.float()
    #
    # input = torch.autograd.Variable(torch.from_numpy(a))
    # target = torch.autograd.Variable(torch.from_numpy(b))
    #
    #
    # print(x)
    # print(y)
    # print(torch.dist(x, y, p=2))
    # print(torch.nn.MSELoss(reduce=False)(x, y))
    # print(torch.nn.MSELoss(reduce=True, size_average=True)(x, y))

    # loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
    # loss = loss_fn(input.float(), target.float())
    # print(loss)
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    # loss = loss_fn(input.float(), target.float())
    # print(loss)

    # model_dict = torch.load('/home/jackzhou/PycharmProjects/CS_CBRSIR/Model_save/test.pkl')['net']
    # for i, (k,p) in enumerate(model_dict.items()):
    #     print(i, k)

    # import numpy as np
    # from sklearn.manifold import TSNE
    # from sklearn import datasets
    # X, y = datasets.make_blobs(n_samples=1000, n_features=512, centers=10, random_state=8)
    #
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    # print(X_embedded.shape)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=1, c=y)
    # plt.show()
    import torch
    print(np.random.randint(0, 8, 1))
    a = torch.ones(size=(2,2))
    b = torch.ones_like(a)
    print(a+b)