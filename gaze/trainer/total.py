import sys,os

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import model
import importlib
import torch
import torch.optim as optim
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
import numpy as np
import torch.nn as nn
import copy
import cv2

def main(config):
    #  ===================>> 开始 <<=================================
    dataloader = importlib.import_module("reader." + config.reader)
    torch.cuda.set_device(config.device) 
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params

    print("===================> 读数据 <===================")
    if data.isFolder:
        data, _ = ctools.readfolder(data)
    dataset = dataloader.loader( data, params.batch_size, shuffle=True, num_workers=0)

    print("===================> 构造模型 <===================")
    net = model.Model()
    net.train()
    net.cuda()

    # Pretrain
    pretrain = config.pretrain
    if pretrain.enable and pretrain.device:
        net.load_state_dict( torch.load(pretrain.path, map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}) )
    elif pretrain.enable and not pretrain.device:
        net.load_state_dict(torch.load(pretrain.path))
        print("已加载预训练模型:",pretrain.path)
    else: print("未预训练")

    print("===================> 优化 <===================")
    optimizer = optim.Adam( net.parameters(), lr=params.lr, betas=(0.9,0.999) )
    scheduler = optim.lr_scheduler.StepLR( optimizer, step_size=params.decay_step, gamma=params.decay)

    if params.warmup:
        scheduler = GradualWarmupScheduler( optimizer, multiplier=1,
                                            total_epoch=params.warmup, after_scheduler=scheduler)
    savepath = os.path.join(save.metapath, save.folder, f"train_model")
    print("savepath:",savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
 
    # =====================================>> 训练 << ====================================
    print("===> Training <===")
    length = len(dataset)
    print("length::",length)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):
                # -------------- 前向传播 -------------
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()
                anno = anno.cuda() 
                loss = net.loss(data, anno)
                # -------------- 反向传播 ------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rest = timer.step()/3600

                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss} " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()
            scheduler.step()

            if epoch % save.step == 0:
                torch.save(net.state_dict(),os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    parser.add_argument('-s', '--train', type=str, help='The source config for training.')
    args = parser.parse_args()
    current_path = os.path.abspath(".")
    args.train = os.path.join(current_path,"/config/train/config_gaze360.yaml")
    config = edict(yaml.load(open(args.train), Loader=yaml.loader.FullLoader))
    main(config.train)







