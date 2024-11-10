import argparse
import logging
import os
import numpy as np
import sys
import torch
from pprint import pprint
from config_vipc import cfg
from run.train import train_net
from run.test import test_net
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#设置GPU设备的排序方式
#PCI_BUS_ID表示按照PCI总线ID排序
#确保我们使用的GPU编号与系统中的GPU编号一致
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE
#指定程序可以使用的GPU设备
#cfg.CONST.DEVICE 这个变量的值通常是一个字符串，表示要使用的GPU的编号。
#例如：os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"表示程序可以使用GPU 0,1,2

'''
假设你有一台机器，上面有3个GPU，编号分别为0、1、2。你想让你的程序只使用编号为1的GPU，并且按照PCI总线顺序来编号GPU，那么你可以这样写代码：
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
'''


def set_seed(seed):
    np.random.seed(seed)#设置numpy的随机种子
    torch.manual_seed(seed)#设置pytorch的随机数生成器随机种子（CPU）
    torch.cuda.manual_seed(seed)#设置pytorch的随机数生成器随机种子（GPU）
    torch.cuda.manual_seed_all(seed)#设置所有 GPU 设备的随机数生成器的种子


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
    args = parser.parse_args()

    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()
    print('cuda available ', torch.cuda.is_available())

    # Print config
    print('Use config:')
    pprint(cfg)

    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if cfg.CONST.WEIGHTS is None:
            raise Exception('Please specify the path to checkpoint in the configuration file!')

        test_net(cfg)

if __name__ == '__main__':
    # Check python version
    seed = 1
    set_seed(seed)
    logging.getLogger('PIL').setLevel(logging.WARNING)#设置PIL库的日志级别
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)#设置日志格式
    main()