#%%

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from dit import DiT_Llama
from PIL import Image
import numpy as np
import os
from tqdm import tqdm


class PreprocessSketchyDataset(Dataset):
    def __init__(self, file_list_path="/root/app/sf/dataset/photo_train.txt", data_dir="/root/app/archived_proj/ZSE-SBIR/datasets/Sketchy",transform=None):
        with open(file_list_path, 'r') as f:
            photo_path_list = f.readlines()
        photo_path_list = [i.strip() for i in photo_path_list]
        
        self.photo_path_list = photo_path_list
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.photo_path_list)
    
    def __getitem__(self, index):
        photo_path = self.photo_path_list[index]
        photo_path, sketch_path = get_sketchy_pair(photo_path)
        
        photo_img = Image.open(os.path.join(self.data_dir, photo_path))
        sketch_img = Image.open(os.path.join(self.data_dir, sketch_path))
        if self.transform:
            photo_img = self.transform(photo_img)
            sketch_img = self.transform(sketch_img)
        return photo_img, sketch_img
        # return photo_img, sketch_img
    
def get_sketchy_pair(photo_path):
    sketch_path = (photo_path.rsplit('.', 1)[0] + '-1.' + photo_path.rsplit('.', 1)[1]).replace('photo', 'sketch').replace('jpg','png')
    return photo_path, sketch_path
    
# def sk_sclale_32():
#%%
ds = PreprocessSketchyDataset()
transform = transforms.Resize((32,32))
for i, (x, c) in tqdm(enumerate(ds)):
    x,c = ds[i]
    img = transform(x)
    sk = transform(c)
    img.save(f'../data/sketchy_32/img/{i}.jpg')
    sk.save(f'../data/sketchy_32/sk/{i}.jpg')

# %%
