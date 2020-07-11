import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import random
from Model.Net import resnet18
import torch


class Generator(object):
    def __init__(self, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_size = 75000
        self.test_size = 80000 - self.train_size

    def center(self, F, PAN_CENTER):
        # MUL_CENTER = torch.zeros(size=(8, 512, 1, 1)).cuda()
        IDX = np.arange(0, 80000)


        with h5py.File('/home/jackzhou/PycharmProjects/CS_CBRSIR/Dataset/DSRSID.mat', 'r') as f:
            # MUL_IMAGES = f["MUL_IMAGES"]
            PAN_IMAGES = f["PAN_IMAGES"]
            LAND_COVER_TYPES = np.array(f["LAND_COVER_TYPES"], np.int).reshape(80000, )-1
            for i in range(8):
                # MUL = MUL_IMAGES[LAND_COVER_TYPES==i]
                # a = np.random.randint(0, MUL.shape[0], 1)
                # MUL = MUL[a]
                # MUL = np.array(MUL)
                # MUL = np.transpose(MUL, axes=(1, 2, 0))
                # MUL = Image.fromarray(MUL)
                # MUL = MUL.resize(size=self.image_size, resample=Image.BICUBIC)
                # MUL = np.expand_dims(np.array(MUL, dtype=np.float).transpose((2, 0, 1)), axis=0)
                # MUL /= 255.
                # MUL = torch.from_numpy(MUL).type(torch.FloatTensor)
                # MUL = MUL.cuda()

                id = IDX[(LAND_COVER_TYPES==i)]
                PAN = PAN_IMAGES[id][:10]
                # a = np.random.randint(0, PAN.shape[0], 1)
                # PAN = PAN[a]
                PAN = np.array(PAN, dtype=np.float)
                PAN = np.repeat(PAN, 4, axis=1) / 255.

                PAN = torch.from_numpy(PAN).type(torch.FloatTensor)
                PAN = PAN.cuda()

                logits, _, _ = F(PAN)
                PAN_CENTER[i] = torch.mean(logits, dim=0)

        return PAN_CENTER

    def generate(self, istrain=True, P = True, M = True):
        with h5py.File('/home/jackzhou/PycharmProjects/CS_CBRSIR/Dataset/DSRSID.mat', 'r') as f:
            self.MUL_IMAGES = f["MUL_IMAGES"]
            self.PAN_IMAGES = f["PAN_IMAGES"]
            self.LAND_COVER_TYPES = np.array(f["LAND_COVER_TYPES"], np.int).reshape(80000, )-1

            MUL = None
            PAN = None
            LABEL = None

            if istrain:
                idx = np.arange(0, 80000)
                random.shuffle(idx)
                train_idx = idx[:self.train_size]
                test_idx = idx[self.train_size:]
                np.save('testidx2.npy', test_idx)
                begin, end = 0, self.batch_size
                while True:
                    if begin < end:
                        id = np.sort(train_idx[begin:end])
                    else:
                        id = np.sort(np.concatenate((train_idx[begin:], train_idx[:end])))
                    if id.shape[0] != self.batch_size:
                        print("size error")
                        raise

                    if P:
                        PAN = self.PAN_IMAGES[id]
                        PAN = np.array(PAN, dtype=np.float)
                        PAN = np.repeat(PAN, 4, axis=1)/255.
                        # plt.figure(1)
                        # plt.imshow(PAN[0].transpose((1, 2, 0)))
                        # plt.show()
                    if M:
                        save = []
                        for i in id:
                            MUL = np.array(self.MUL_IMAGES[i])
                            MUL = np.transpose(MUL, axes=(1, 2, 0))
                            MUL = Image.fromarray(MUL)
                            MUL = MUL.resize(size=self.image_size, resample=Image.BICUBIC)
                            MUL = np.expand_dims(np.array(MUL, dtype=np.float).transpose((2, 0, 1)), axis=0)
                            save.append(MUL)
                        MUL = np.concatenate(save, axis=0)/255.
                        # plt.figure(2)
                        # plt.imshow(MUL[0].transpose((1, 2, 0)))
                        # plt.show()

                    if M or P:
                        LABEL = self.LAND_COVER_TYPES[id]

                    begin = end
                    end = (end+self.batch_size) % 75000


                    yield PAN, MUL, LABEL
            elif not istrain:
                test_idx = np.load("/home/jackzhou/PycharmProjects/CS_CBRSIR/Model/testidx2.npy")
                begin, end = 0, self.batch_size
                while True:
                    if begin < end:
                        id = np.sort(test_idx[begin:end])
                    else:
                        id = np.sort(np.concatenate((test_idx[begin:], test_idx[:end])))
                    if id.shape[0] != self.batch_size:
                        print(id.shape[0], begin, end)
                        print("size error")
                        raise
                    if P:
                        PAN = self.PAN_IMAGES[id]
                        PAN = np.array(PAN, dtype=np.float)
                        PAN = np.repeat(PAN, 4, axis=1)/255.

                    if M:
                        save = []
                        for i in id:
                            MUL = np.array(self.MUL_IMAGES[i])
                            MUL = np.transpose(MUL, axes=(1, 2, 0))
                            MUL = Image.fromarray(MUL)
                            MUL = MUL.resize(size=(256, 256), resample=Image.BICUBIC)
                            MUL = np.expand_dims(np.array(MUL, dtype=np.float).transpose((2, 0, 1)), axis=0)
                            save.append(MUL)
                        MUL = np.concatenate(save, axis=0)/255.

                    if M or P:
                        LABEL = self.LAND_COVER_TYPES[id]

                    begin = end
                    end = (end + self.batch_size) % 5000
                    yield PAN, MUL, LABEL






# if __name__ == "__main__":
#     # load_dataset()
#     gen = Generator(128, (256, 256)).generate(istrain=True, P=True, M=True)
#     gen2 = Generator(32, (256, 256)).generate(False)
#
#     for i, (PAN, MUL, LABEL) in enumerate(gen):
#         print(PAN, MUL)
#         print(PAN.shape, LABEL.shape)
#         print(LABEL)
    # for i in range(1000):
    #     PAN, MUL, LABEL = next(gen)
    #     print(PAN.shape, MUL.shape, LABEL.shape)
    #     print(LABEL)
    #     PAN, MUL, LABEL = next(gen2)
    #     print(PAN.shape, MUL.shape, LABEL.shape)
    #     print(LABEL)

    # for i, (PAN, MUL, LABEL) in enumerate(gen):
    #     print(PAN.shape, MUL.shape, LABEL.shape)
    #     print(LABEL)