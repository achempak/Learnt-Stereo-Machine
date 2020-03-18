import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import os
import time

from kaolin.metrics.point import chamfer_distance

from config import SHAPENET_IM, SHAPENET_VOX
from shapenet_pytorch import ShapeNetDataset
from lsm import LSM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: "+str(device))

### TRAINING PARAMETERS
epochs = 10
lr = 0.0001
nvox = 32
batch_size = 10
nviews = 4

### DATALOADING
vox_dir = SHAPENET_VOX[nvox]
im_dir = SHAPENET_IM
split_file = './splits.json'

categories = ['sofa']
dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=True, categories=categories)
train_batch_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
test_dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=False, categories=categories)
test_batch_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

### CHECKPOINTS
if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')
checkpoint_path = './checkpoints/experiment_{}'.format(str(int(time.time())))
os.mkdir(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path,'best_model.pth')

# LSM model
lsm = LSM()
#loss_func = torch.nn.BCELoss()
loss_func = chamfer_distance
optimizer = torch.optim.Adam(lsm.parameters(), lr=lr)


def train(lsm, epochs, lr, train_batch_loader, test_batch_loader):
    best_loss = 100000000.0
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_batch_loader):
            imgs, vox, K, R = batch
            imgs = imgs.type(torch.FloatTensor).to(device)
            vox = vox.type(torch.FloatTensor).to(device)
            K = K.type(torch.FloatTensor).to(device)
            R = R.type(torch.FloatTensor).to(device)
            K = K.view(-1, 3, 3)
            R = R.view(-1, 3, 4)

            optimizer.zero_grad()
            vox_pred = lsm(imgs, K, R)
            # print("vox_pred dims: "+str(vox_pred.shape))
            # print("vox dims: "+str(vox.shape))
            loss = torch.Tensor([0]).to(device)
            while len(vox_pred.shape) < 5:
                vox_pred.unsqueeze(0)
            while len(vox.shape) < 5:
                vox.unsqueeze(0)
            size = vox_pred.shape[0]
            for j in range(size):
                loss += loss_func(vox_pred[j].squeeze().view((nvox, -1)), vox[j].squeeze().view((nvox, -1)))
            loss /= size
            #loss = loss_func(vox_pred, vox)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("[{}/{}] epoch, [{}/{}] batch: Loss = {}".format(epoch, epochs, i, len(train_batch_loader), loss.item()))

        train_loss = running_loss/len(train_batch_loader)
        test_loss = test(test_batch_loader, lsm)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        np.save(os.path.join(checkpoint_path, 'losses.npy'), np.stack((train_losses, test_losses)))
        print("[{}/{}] Training loss = {}, Test Loss = {}".format(epoch, epochs, train_loss, test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(lsm.state_dict(), checkpoint_file)

def test(test_batch_loader, lsm):
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(test_batch_loader):
            imgs, vox, K, R = batch
            imgs = imgs.type(torch.FloatTensor).to(device)
            vox = vox.type(torch.FloatTensor).to(device)
            K = K.type(torch.FloatTensor).to(device)
            R = R.type(torch.FloatTensor).to(device)
            K = K.view(-1, 3, 3)
            R = R.view(-1, 3, 4)

            vox_pred = lsm(imgs, K, R)
            loss = torch.Tensor([0]).to(device)
            while len(vox_pred.shape) < 5:
                vox_pred.unsqueeze(0)
            while len(vox.shape) < 5:
                vox.unsqueeze(0)
            size = vox_pred.shape[0]
            for j in range(size):
                loss += loss_func(vox_pred[j].squeeze().view((nvox, -1)), vox[j].squeeze().view((nvox, -1)))
            loss /= size
            running_loss += loss.item()

    return running_loss/len(test_batch_loader)


if __name__ == "__main__":
    train(lsm, epochs, lr, train_batch_loader, test_batch_loader)
