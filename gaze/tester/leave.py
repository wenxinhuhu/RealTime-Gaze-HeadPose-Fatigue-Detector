import os, sys
# base_dir = os.getcwd()
# print(base_dir)
# sys.path.insert(0, os.path.dirname(base_dir))
sys.path.insert(0, "/trainer")

from model import Model
import importlib
import yaml
import torch
from easydict import EasyDict as edict
import ctools, gtools
import argparse
import torch.backends.cudnn as cudnn
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import cv2, copy

def main(train, test):

    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)
    cudnn.benchmark = True

    data = test.data
    load = test.load

    data, folder = ctools.readfolder(data, [test.person])
    testname = folder[test.person]

    dataset = reader.loader(data, 500, num_workers=0, shuffle=True)
    modelpath = os.path.join(train.save.metapath,train.save.folder, f'checkpoint/{testname}')  # 模型路径
    logpath = os.path.join(train.save.metapath,train.save.folder, f'{test.savename}/{testname}')  # test日志文件路径

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    print("=============================> 测试 <==============================")
    begin = load.begin_step  #10
    end = load.end_step  #50
    step = load.steps  #10

    for saveiter in range(begin, end+step, step):
        print(f"Test {saveiter}")
        # ----------------------Load Model------------------------------
        net = Model() #实例化模型
        statedict = torch.load(
                                os.path.join(modelpath, f"Iter_{saveiter}_{train.save.model_name}.pt"),
                                map_location={f"cuda:{train.device}":f"cuda:{test.device}"}
                              )
        net.cuda()
        net.load_state_dict(statedict)
        net.eval()


        accs = 0
        count = 0

        # -----------------------打开日志文件--------------------------------
        logname = f"{saveiter}.log"
        outfile =  open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")


        # -------------------------Testing---------------------------------
        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()
                names = data["name"]
                gts = label
                gazes = net(data)

                for k, gaze in enumerate(gazes):
                    gaze = gaze.cpu().detach().numpy()  # 预测值
                    gt = gts.numpy()[k]   # 真实值

                    count += 1
                    accs += gtools.angular(gtools.gazeto3d(gaze),  # 角度转换
                                           gtools.gazeto3d(gt))  # 计算角度误差
                    name = [names[k]]
                    # print("name:::::::::",name)
                    gaze = [str(u) for u in gaze]
                    # print("gaze::::::",gaze)
                    gt = [str(u) for u in gt]
                    # print("gt::::::",gt)
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    # print("log:::::::::",log)
                    # print("============================")
                    outfile.write(" ".join(log) + "\n")

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()

if __name__ == "__main__":
    for i in list(range(15)):
        parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')
        parser.add_argument('-s', '--source', type=str,help = 'config path about training')
        parser.add_argument('-t', '--target', type=str, help = 'config path about test')
        parser.add_argument('-p', '--person', type=int,help = 'the num of subject for test')
        args = parser.parse_args()

        current_path = os.path.abspath(".")
        args.source = os.path.join(current_path,"/config/train/config_mpii.yaml")
        args.target = os.path.join(current_path,"/config/test/config_mpii.yaml")
        # Read model from train config and Test data in test config.
        train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))
        test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))
        test_conf = test_conf.test
        test_conf.person = args.person

        test_conf.person = i
        main(train_conf.train, test_conf)

