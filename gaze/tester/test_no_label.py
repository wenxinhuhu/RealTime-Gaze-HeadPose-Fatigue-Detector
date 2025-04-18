# tester/test_no_label.py
import os, sys, importlib, yaml, argparse, torch
from easydict import EasyDict as edict

sys.path.insert(0, os.getcwd())   # 保证能 import model / reader

from model import Model
import ctools

def main(train_cfg, test_cfg):
    # ---------- 1. 读数据 ----------
    reader = importlib.import_module(test_cfg.reader)
    torch.cuda.set_device(test_cfg.device)

    data = test_cfg.data            # EasyDict
    if not hasattr(data, "label"):  # 没有标签时给一个空串，省得后面误用
        data.label = ""

    if data.isFolder:               # 只有目录式标签才需要 readfolder
        data, _ = ctools.readfolder(data)

    print(f"==> Test images root: {data.image} <==")

    dataset = reader.loader(data, batch=128, shuffle=False, num_workers=4)
    print(f"Loaded {len(dataset)} images.")
    
    # ---------- 2. 读模型 ----------
    mp = train_cfg.save             # 取出保存路径信息
    model_dir = os.path.join(mp.metapath, mp.folder, "train_model")
    log_dir   = os.path.join(mp.metapath, mp.folder, test_cfg.savename)
    os.makedirs(log_dir, exist_ok=True)

    for step in range(test_cfg.load.begin_step,
                      test_cfg.load.end_step + 1,
                      test_cfg.load.steps):

        w_path = os.path.join(model_dir, f"Iter_{step}_{mp.model_name}.pt")
        print(f"[+] Loading checkpoint {w_path}")

        net = Model().cuda()
        net.load_state_dict(torch.load(w_path,
                         map_location={f"cuda:{train_cfg.device}": f"cuda:{test_cfg.device}"}))
        net.eval()

        out_path = os.path.join(log_dir, f"{step}_gaze_pred.txt")
        with open(out_path, "w") as f, torch.no_grad():
            f.write("name gaze\n")
            for batch in dataset:
                for key in batch:
                    if key != "name":
                        batch[key] = batch[key].cuda()

                names = batch["name"]
                gazes = net(batch).cpu().numpy()

                for name, gaze in zip(names, gazes):
                    gaze_str = ",".join(map(str, gaze))
                    f.write(f"{name} {gaze_str}\n")

        print(f"[✓] Saved predictions to {out_path}")

# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=r"E:\multi_eye\RealTime-Gaze-HeadPose-Fatigue-Detector\gaze\config\test\config_myset.yaml")
    args = parser.parse_args()

    cfg = edict(yaml.safe_load(open(args.config, encoding="utf-8")))
    main(cfg.train, cfg.test)
