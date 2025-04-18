import sys, os
from model import Model  #net = Model()

base_dir = os.getcwd()
# print(base_dir)
sys.path.insert(0, os.path.dirname(base_dir))

# import model   # net = model.Model()
import importlib
import torch
import torch.optim as optim
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
#from trainer import model
import cv2
import copy
import numpy as np
import torch.nn as nn


#用于交叉验证

def main(configx,person_num):
    # ===============================> 开始 <================================
    dataloader = importlib.import_module(config.reader)  # config.reader = reader  导入reader.reader下的包
    torch.cuda.set_device(config.device)  # pytorch是cpu版的，不能使用GPU,config.device需要设置为-1
    cudnn.benchmark = True  # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

    data = config.data
    save = config.save
    params = config.params
    config.person = person_num

    # 读数据  data是剔除N份数据中的1份，保留N-1份   folder：完整的N份数据
    data, folder = ctools.readfolder(data, [config.person], reverse=True)
    savename = folder[config.person]  # 取出之前剔除的那一份数据
    dataset = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=0)  # 读数据

    print("=====================> 构建模型 <=====================")
    net = Model()
    net.train()  # 启用 BatchNormalization 和 Dropout
    # net = nn.DataParallel(net) 用多个GPU来加速训练
    net.cuda()

    # 预训练
    # pretrain = config.pretrain
    # if pretrain.enable and pretrain.device:
    #     net.load_state_dict(
    #                          torch.load(pretrain.path,map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"})
    #                        )
    # elif pretrain.enable and not pretrain.device:
    #     net.load_state_dict( torch.load(pretrain.path) )

    print("=====================> 构建优化器 <=====================")
    optimizer = optim.Adam(net.parameters(), lr=params.lr, betas=(0.9, 0.95))
    # 随着迭代次数的增加，学习率逐步衰减。 每step_size个epoch后lr衰减一次,变为原来的gamma倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.decay_step, gamma=params.decay)

    if params.warmup:
        scheduler = GradualWarmupScheduler( optimizer, multiplier=1,
                                            total_epoch=params.warmup, after_scheduler=scheduler)

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint/{savename}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("=====================> 训练 <=====================")
    length = len(dataset)  # batch的数量
    print("length:", length)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad()  # 梯度置零
    optimizer.step()  #更新参数
    scheduler.step()

    p_num = sum([p.numel() for p in net.parameters()])
    print("Number of parameter: %.2fM" % (p_num / 1e6))  # 统计模型参数

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')   #调整config文件的格式
        for epoch in range(1, params.epoch + 1):    #循环的epoch数量
            for i, (data, anno) in enumerate(dataset):
                # ------------------前向传播--------------------
                data["face"] = data["face"].cuda()  #torch.Size([64, 3, 224, 224])
                anno = anno.cuda() # torch.Size([64, 2])
                """
                  data数据结构：{ 'face' : ....
                                  'name': ['day29/0018.jpg',........]
                                }      
                """
                loss = net.loss(data, anno)

                # -----------------backward--------------------
                optimizer.zero_grad()  #梯度置零
                loss.backward()
                optimizer.step()
                rest = timer.step() / 3600

                # -----------------loger----------------------
                if i % 20 == 0: # 迭代次数/总迭代次数  #每一轮中第i个batch/总的batch数  #损失值  #学习率  #剩余时间
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " + \
                          f"loss:{loss}" + \
                          f" lr:{ctools.GetLR(optimizer)} " + \
                          f"rest_time:{rest:.2f}h"
                    print(log)
                    outfile.write(log + "\n")  #写入日志文件
                    sys.stdout.flush() # 显示地让缓冲区的内容输出
                    outfile.flush()  #刷新缓冲区的，将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入

            scheduler.step() # 参数更新

            if epoch % save.step == 0:
                torch.save(
                            net.state_dict(),  #保存模型参数
                            os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pt")  #保存的路径
                          )


if __name__ == "__main__":
    for i in list(range(15)):
        parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')#    本pytorch是cpu版的，不能使用GPU,config.device需要设置为-1
        parser.add_argument('-s', '--train', type=str, help='The source config for training.')
        parser.add_argument('-p', '--person', type=int,help='The tested person.')
        args = parser.parse_args()
        current_path = os.path.abspath(".")
        args.train = os.path.join(current_path,"config/train/config_mpii.yaml")

        config = edict(yaml.load(open(args.train), Loader=yaml.loader.FullLoader))
        config = config.train
        main(config,i)

