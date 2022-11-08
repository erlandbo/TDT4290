from torch.utils.data import Dataset 
import numpy as np
from tqdm import tqdm
import os
from .spectogram_helper import spec_to_image, get_melspectrogram_db

class AudioRegressionDataset(Dataset):
    def __init__(self, base, df, in_featr, out_featr):
        self.df = df
        self.data = []
        self.labels = []
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = os.path.join(base,row[in_featr])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
            self.labels.append(row[out_featr])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class AudioClassificationDataset(Dataset):
    def __init__(self, base, df, in_featr, out_featr):
        self.df = df
        self.data = []
        self.labels = []
        self.categories = sorted(df[out_featr].unique())
        self.category_to_label = {category: i for i, category in enumerate(self.categories)}
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            file_path = os.path.join(base, row[in_featr])
            self.data.append(spec_to_image(get_melspectrogram_db(file_path))[np.newaxis,...])
            self.labels.append(self.category_to_label[row[out_featr]])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]