import os, sys

sys.path.insert(0, "/trainer")

from model import Model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

def main(train, test):

    # =================================> 开始 <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)
    data = test.data
    load = test.load

    # ===============================> 读数据 <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 128, num_workers=4, shuffle=False)

    modelpath = os.path.join(train.save.metapath,train.save.folder, f"train_model/")# 模型路径
    logpath = os.path.join(train.save.metapath,train.save.folder, f"{test.savename}")# test日志文件路径

    if not os.path.exists(logpath):
        os.makedirs(logpath)
    # =============================> 测试 <=============================
    begin = load.begin_step
    end = load.end_step
    step = load.steps
    for saveiter in range(begin, end+step, step):
        print(f"Test {saveiter}")
        net = Model()
        statedict = torch.load(os.path.join(modelpath, f"Iter_{saveiter}_{train.save.model_name}.pt"),
                                map_location={f"cuda:{train.device}": f"cuda:{test.device}"} )

        net.cuda()
        net.load_state_dict(statedict)
        net.eval()

        length = len(dataset)
        accs = 0
        count = 0

        logname = f"{saveiter}.log"
        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")
        

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names = data["name"]
                gts = label.cuda()
                gazes = net(data)
                for k, gaze in enumerate(gazes):
                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]
                    count += 1                
                    accs += gtools.angular(gtools.gazeto3d(gaze),gtools.gazeto3d(gt))
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
    parser.add_argument('-s', '--source', type=str,help = 'config path about training')
    parser.add_argument('-t', '--target', type=str, help = 'config path about test')
    parser.add_argument('-p', '--person', type=int,help = 'the num of subject for test')
    args = parser.parse_args()

    current_path = os.path.abspath(".")
    args.source = os.path.join(current_path,"config/train/config_gaze360.yaml")
    args.target = os.path.join(current_path,"config/test/config_gaze360.yaml")
    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))
    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))
    test_conf = test_conf.test
    test_conf.person = args.person
    main(train_conf.train, test_conf)


 
