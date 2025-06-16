import torch
import os
import time
from model_pre import Network
from decoder.utils.utils import *
import wandb
import argparse
import logging
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from datetime import datetime
from testv1 import test_net

import open3d as o3d
import os
import numpy as np
from PIL import Image

from Symm.symm import SymmCompletion, SymmCompletion2, SymmCompletion3

import config_vipc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


cfg = config_vipc.cfg

model=SymmCompletion3(cfg)


print(f"Total Parameters: {count_parameters(model)/1e6:.2f}M")