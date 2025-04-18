# reader/frames.py
import os, cv2, torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FramesLoader(Dataset):
    def __init__(self, root):             # root = data.image
        self.root = root
        self.files = sorted(os.listdir(root))   # 只按文件名排序
        self.tf = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.root, fname))
        img = self.tf(img)

        data = edict()
        data.face = img           # tester/total.py 只用到 data["face"] 和 data["name"]
        data.name = fname
        return data               # 没有 label

def loader(source, batch, shuffle, num_workers):
    ds = FramesLoader(source.image)       # 这里只用 data.image
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers)
