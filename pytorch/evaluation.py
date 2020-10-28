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

g = GraspNet('/ssd1/graspnet/', camera='kinect', split='test')

height = 0.02
depth_base = 0.02
grasp_depth = 0.02
grasp_width = 0.06
num_sample = 2000
DUMP_DIR = './dump/'
sceneIds = g.getSceneIds()

input_channels = 3
model = Net(input_channels)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load('model_new_2.pwf'))
model.eval()


def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

for sceneId in range(100,190):
    for i in tqdm(range(256)):
        t1 = time.time()
        cloud = g.loadScenePointCloud(sceneId, 'kinect', i, remove_outlier=True, align=False, format = 'open3d')
        fullcloud = g.loadScenePointCloud(sceneId, 'kinect', i, remove_outlier=False, align=False, format = 'open3d')
        downpc = cloud.voxel_down_sample(voxel_size=0.005)
        sparsepc = cloud.voxel_down_sample(voxel_size=0.03)
        points = np.array(downpc.points).astype(np.float32)
        rgbs = np.array(downpc.colors).astype(np.float32)
        t2 = time.time()
        print("time read data: ",t2-t1)
        normals = estimate_normals(points, k=10, align_direction=False, ret_cloud=False)
        #frames = estimate_darboux_frame(points, normals, dist_thresh=0.01)
        idx = np.random.choice(len(points), num_sample)
        grasp_points = points[idx]
        grasp_normals = normals[idx]
        grasp_frames = estimate_darboux_frame(grasp_points, grasp_normals, points, normals, dist_thresh=0.01)
        
	#grasp_frames = frames[idx]
        t3 = time.time()
        print("process frame: ",t3-t2)
        
        points_centered = points[np.newaxis,:,:] - grasp_points[:,np.newaxis,:]

        targets = np.matmul(points_centered, grasp_frames) #(num_sample, num_point, 3)
        t4 = time.time()
        print("transform points: ",t4-t3)

        feats = []
        grasp_group = []

        for j, ind in enumerate(range(targets.shape[0])):
            target = targets[ind]
            mask1 = ((target[:,2]>-height) & (target[:,2]<height))
            mask2 = ((target[:,0]<grasp_depth) & (target[:,0]>-depth_base))
            mask3 = ((target[:,1]>-grasp_width/2) & (target[:,1]<grasp_width/2))
            mask = (mask1 & mask2 & mask3)
            pc = target[mask]
            if len(pc) == 0:
                continue
            pc = estimate_normals(pc, k=20, align_direction=True, ret_cloud=True)
            pc.colors = o3d.utility.Vector3dVector(rgbs[mask])
            # o3d.io.write_point_cloud('res/'+str(i)+'_'+str(j)+'.ply', pc, write_ascii=False, compressed=True)

            img = transform_cloud_to_image(pc)
            # cv2.imwrite('res/'+str(i)+'_'+str(j)+'.jpg', img)

            feats.append(img * 1/255.0) 
            grasp_group.append(flat([1, grasp_width, height, grasp_depth, grasp_frames[j].reshape(9).tolist(), grasp_points[j].reshape(3).tolist(), -1]))
        t5 = time.time()
        print("crop points: ",t5-t4)

        grasp_group = np.array(grasp_group)
        print(grasp_group)
        mfcdetector = ModelFreeCollisionDetector(np.array(fullcloud.points).astype(np.float32), voxel_size=0.005)
        collision_mask = mfcdetector.detect(GraspGroup(grasp_group), approach_dist=0.03, collision_thresh=0.01)

        t6 = time.time()
        print("collision checking: ",t6-t5)

        feats = np.array(feats)
        valid_feats = feats[~collision_mask]
        print(collision_mask.shape)
        print(valid_feats.shape)

        with torch.no_grad():
            
            inputs = torch.from_numpy(np.array(valid_feats).astype(np.float32)).permute(0, 3, 1, 2).to(device)
            outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            scores = outputs.data[:,1]

        final_grasps = grasp_group[~collision_mask]
        print(final_grasps.shape)
        final_grasps[:,0] = scores.detach().cpu().numpy()
        print(final_grasps.shape)

        if not os.path.exists(os.path.join(DUMP_DIR, 'scene_'+str(sceneId).zfill(4))):
            os.makedirs(os.path.join(DUMP_DIR, 'scene_'+str(sceneId).zfill(4)))
        print(np.array(sparsepc.points).shape)
        print(np.array(sparsepc.colors).shape)
        np.savez_compressed(os.path.join(DUMP_DIR,  'scene_'+str(sceneId).zfill(4), str(i%256).zfill(4)+'.npz'), clouds=np.array(sparsepc.points), colors=np.array(sparsepc.colors), preds=np.array(final_grasps))
        print(1111)

ge_k = GraspNetEval(root = '/ssd1/graspnet/', camera = 'kinect', split = 'test')
print('Evaluating kinect')
res, ap = ge_k.eval_all(DUMP_DIR, proc = 24)

def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    print('Testing the network on the test data ...')

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    accuracy = 100.0 * float(correct) / float(total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))

    return accuracy

if len(sys.argv) < 4:
    print('ERROR: Not enough input arguments!')
    print('Usage: python continue_train_net.py pathToTrainingSet.json pathToTestingSet.json old_model.pwf')
    exit(-1)

with open(sys.argv[1], 'r') as db:
    t_dict = json.load(db)
    num_train = len(t_dict['images'])

print('Have', num_train, 'total training examples')
num_epochs = 3
max_in_memory = 80000
print_step = 250
repeats = 1
early_stop_loss = 0.0000001
start_idx = 0
end_idx = max_in_memory
iter_per_epoch = int(np.ceil(num_train / float(max_in_memory)))
indices = np.arange(0, num_train, max_in_memory)
indices = list(indices) + [num_train]
print('iter_per_epoch:', iter_per_epoch)
print(indices)

# Use GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the test data.
print('Loading test data ...')
test_set = JsonDataset(sys.argv[2])
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True)

# Create the network.
#input_channels = test_set.images.shape[-1]
input_channels = 3
#net = NetCCFFF(input_channels)
net = Net(input_channels)

print('Copying network to GPU ...')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)
net.to(device)

net.load_state_dict(torch.load(sys.argv[3]))
print(net)

# Define the loss function and optimizer.
LR = 0.0001
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)

optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
accuracy = eval(net, test_loader, device)
accuracies = []
accuracies.append(accuracy)

early_stop = False
losses = []
loss = None
print('Training ...')

for epoch in range(num_epochs):
    print('epoch: %d/%d' % (epoch + 1, num_epochs))
    net.train()

    for param_group in optimizer.param_groups:
        print('learning rate:', param_group['lr'])

    for j in range(iter_per_epoch):
        print('iter: %d/%d' % (j + 1, iter_per_epoch))
        print('Loading data block [%d, %d] ...' % (indices[j], indices[j + 1]))
        dset = []
        train_loader = []
        dset = JsonDataset(sys.argv[1], indices[j], indices[j + 1])
        #train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)
        train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True)

        running_loss = 0.0

        for r in range(repeats):
            if r > 1:
                print('repeat: %d/%d' % (r + 1, repeats))
            for i, data in enumerate(train_loader):
                loss = train(net, criterion, optimizer, data, device)

                # print statistics
                running_loss += loss.item()
                if i % print_step == print_step - 1:
                    print('epoch: {}, batch: {}, loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / print_step))
                    losses.append(running_loss)
                    running_loss = 0.0
            # Evaluate the network on the test dataset.
            accuracy = eval(net, test_loader, device)
            accuracies.append(accuracy)
            model_path = 'model_' + str(accuracy) + '.pwf'
            torch.save(net.state_dict(), model_path)
            net.train()
            if early_stop:
                break
        if early_stop:
            break
    if early_stop:
        break

    # Evaluate the network on the test dataset.
    accuracy = eval(net, test_loader, device)
    accuracies.append(accuracy)
    model_path = 'model_' + str(accuracy) + '.pwf'
    torch.save(net.state_dict(), model_path)
    #scheduler.step()

print('Finished Training')

model_path = 'model_new.pwf'
torch.save(net.state_dict(), model_path)

with open('loss_stats_new.txt', 'w') as f:
    for l in losses:
        f.write("%s\n" % str(l))
with open('accuracy_stats_new.txt', 'w') as f:
    for a in accuracies:
        f.write("%s\n" % str(a))
