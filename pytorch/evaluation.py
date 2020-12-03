	# Use tensors to speed up loading data onto the GPU during training.
import numpy as np
import open3d as o3d
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys
from graspnetAPI import GraspNet, GraspNetEval, GraspGroup
#torch.multiprocessing.set_start_method('spawn')
from grasptoolbox.grasp_sampling import estimate_normals, estimate_darboux_frame, transform_cloud_to_image
from grasptoolbox.collision_detection import ModelFreeCollisionDetector

from json_dataset import JsonDataset
from network import Net, NetCCFFF
from tqdm import tqdm
import os
import time
import cv2


DUMP_DIR = './dump/'


ge_k = GraspNetEval(root = '/home/minghao/graspnet/', camera = 'kinect', split = 'test_seen')
print('Evaluating kinect')
ap = ge_k.eval_scene(100, DUMP_DIR)
print('AP: ',np.array(ap).mean())

