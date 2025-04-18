import os
import cv2
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno


def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

#
def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno
#
#
def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno


def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.name = line[0]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    # mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    # mapping.ethtrain = Decode_ETH
    # mapping.rtgene = Decode_RTGene
    return mapping

# print(long_substr('12g34567','12gh34567')) 返回5
def long_substr(str1, str2):  #  返回str1中包含多少个与str2相同的子字符串，返回最大的子字符串的长度
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1) - i + 1):
            if j > len(substr) and (str1[i:i + j] in str2):
                substr = str1[i:i + j]
    return len(substr)

#  Get_Decode函数将数据集名称作为输入，并返回相应的注释解码函数。
def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))]
    return mapping[key]


class trainloader(Dataset):
    def __init__(self, dataset):
        #读取源数据
        self.data = edict()  #相当于是生成一个空的字典
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):  #判断数据集标签是否是 列表形式
            for i in dataset.label:  #遍历标签数据
                with open(i) as f:
                    line = f.readlines()
                if dataset.header:
                    line.pop(0)  # .pop(0) 删除第一个元素
                self.data.line.extend(line) #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值

        else:
            with open(dataset.label) as f:
                self.data.line = f.readlines()
            if dataset.header: self.data.line.pop(0)

        # build transforms
        self.transforms = transforms.Compose([ transforms.ToTensor() ])

    def __len__(self):
        return len(self.data.line)  #返回数据集中样本数量

    def __getitem__(self, idx):  #返回预处理图像及标签数据
        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        img = cv2.imread(os.path.join(self.data.root, anno.face))  #os.path.join()函数用于路径拼接文件路径
        img = self.transforms(img)  #将图片转换成 Tensor

        label = np.array(anno.gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        data = edict()
        data.face = img
        data.name = anno.name

        return data, label


def loader(source, batch_size, shuffle=True, num_workers=0):
    dataset = trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

