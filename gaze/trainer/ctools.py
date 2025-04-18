import numpy as np
import sys
import time
import os
import json
from easydict import EasyDict as edict

class TimeCounter:
    # Create an time counter.
    # To count the rest time.

    # Input the total times.
    def __init__(self, total):
      self.total = total
      self.cur = 0
      self.begin = time.time()

    def step(self):
      end = time.time() 
      self.cur += 1
      used = (end - self.begin)/self.cur
      rest = self.total - self.cur

      return np.max(rest * used, 0)
         

def readfolder(data, specific=None, reverse=False):
    folders = os.listdir(data.label) # 返回指定的文件夹包含的文件或文件夹的名字的列表
    folders.sort()  # 排序
    folder = folders
    if specific is not None:
        if reverse:
            num = np.arange(len(folders)) # 生成长为folders的numpy数组
            specific = list(filter(lambda x: x not in specific, num))  #filter()函数用于过滤序列，返回由符合条件元素组成的新列表。
        folder = [folders[i] for i in specific]    # 保存specific中指定的那份数据

    data.label = [os.path.join(data.label, j) for j in folder]

    return data, folders


def DictDumps(content):
    return json.dumps(content,  ensure_ascii=False, indent=4)  #用dumps将python编码成json字符串
# ensure_ascii ：默认值为 True，会将所有输入的非 ASCII 字符转义输出，如果值为 False，会将输入的非 ASCII 字符原样输出

# indent:默认值为 None。选择最紧凑的表达。
#     如果 indent 是一个非负整数或者字符串，那么 JSON 数组元素和对象成员会被美化输出为该值指定的缩进等级。
#     如果缩进等级为零、负数或者 “”，则只会添加换行符。
#     当 indent 为一个正整数时会让每一层缩进同样数量的空格；
#     如果 indent 是一个字符串如换行符、制表符 ( “\n”、 “\t”) 等，那么这个字符串会被用于每一层



def GetLR(optimizer):
    LR = optimizer.state_dict()['param_groups'][0]['lr']
    return LR



