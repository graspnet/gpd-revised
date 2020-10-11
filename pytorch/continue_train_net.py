	# Use tensors to speed up loading data onto the GPU during training.

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys

#torch.multiprocessing.set_start_method('spawn')

from json_dataset import JsonDataset
from network import Net, NetCCFFF

def train(model, criterion, optimizer, data, device):
    # Get the inputs and transfer them to the CPU/GPU.
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Reset the parameter gradients.
    optimizer.zero_grad()

    # Forward + backward + optimize.
    outputs = model(inputs)
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    return loss

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
