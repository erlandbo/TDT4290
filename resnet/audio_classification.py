"""
Author Erland Brandser Olsson, erlandbo.olsson@gmail.com
Parts of the code is taken and inspired by https://github.com/hasithsura/Environmental-Sound-Classification
"""


import numpy as np
import librosa
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import soundfile as sf
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
        hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(spec, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

class OurDataSet(Dataset):
    def __init__(self, base_path, df, in_featr, out_featr):
        self.df = df
        self.data = []
        self.labels = []
        self.categories = sorted(df[out_featr].unique())
        self.category_to_label = {category: i for i, category in enumerate(self.categories)}
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            file_path = os.path.join(base_path, row[in_featr])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
            self.labels.append(self.category_to_label[row[out_featr]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    
    
def setlr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def lr_decay(optimizer, epoch, learning_rate):
    if epoch%10==0:
        new_lr = learning_rate / (10**(epoch//10))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer

def train(model, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, device,learning_rate, change_lr=None):
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        batch_losses=[]
        if change_lr:
            optimizer = change_lr(optimizer, epoch,learning_rate)

        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)
        
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')
        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        for i, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')


def main(num_classes=3):
    path = str(Path.cwd())
    train_path = path + "/train/"
    valid_path = path + "/valid/" 
    train_data = pd.read_csv(train_path + "train.csv")
    valid_data = pd.read_csv(valid_path + "valid.csv")
    if num_classes == 3:
        train_data["class"] = train_data["class_1"]
        valid_data["class"] = valid_data["class_1"]
    else:
        train_data["class"] = train_data["class_2"]
        valid_data["class"] = valid_data["class_2"]
    train_data = OurDataSet(train_path, train_data, in_featr='filename', out_featr='class')
    valid_data = OurDataSet(valid_path, valid_data, in_featr='filename', out_featr='class')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')     
    resnet_model = resnet34(weights="DEFAULT") #resnet34(pretrained=True)
    resnet_model.fc = nn.Linear(512,3) if num_classes==3 else nn.Sequential(nn.Linear(512,512), nn.Linear(512,512), nn.Linear(512,5))  # 512, 50
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet_model = resnet_model.to(device)
    
    learning_rate = 2e-4
    optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
    epochs = 50
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    train(resnet_model, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, device, lr_decay)
    
    tl = np.asarray(train_losses).ravel()
    vl = np.asarray(valid_losses).ravel()
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(tl)
    plt.legend(['Train Loss'])
    plt.subplot(1,2,2)
    plt.plot(vl,'orange')
    plt.legend(['Valid Loss'])
    plt.show()
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--class", default=3, type=int, help="Number of classes in ANN model")
    args = vars(parser.parse_args())
    num_classes = args["class"]
    assert num_classes == 3 or num_classes==5, "Invalid number of classes"
    main(num_classes=num_classes)