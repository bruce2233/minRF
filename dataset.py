from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
from dit import DiT_Llama
from PIL import Image
import numpy as np
import os, random
from glob import glob


class SketchyDataset(Dataset):
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
        return sketch_img, 0
        # return photo_img, sketch_img

class Sketchy32Dataset(Dataset):
    def __init__(self, data_dir="/root/app/minRF/data/sketchy_32",transform=None):
        
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(glob(f"{self.data_dir}/img/*"))
    
    def __getitem__(self, index):
        photo_img = Image.open(f"{self.data_dir}/img/{index}.jpg")
        sketch_img = Image.open(f"{self.data_dir}/sk/{index}.jpg")
        if self.transform:
            photo_img = self.transform(photo_img)
            sketch_img = self.transform(sketch_img)
            # sketch_photo_chunk = torch.stack([sketch_img, photo_img])
        return (sketch_img, photo_img), random.randint(0,9)
        # return photo_img, sketch_img
        
class Sketchy32NocondDataset(Dataset):
    def __init__(self, data_dir="/root/app/minRF/data/sketchy_32",transform=None):
        
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(glob(f"{self.data_dir}/img/*"))
    
    def __getitem__(self, index):
        photo_img = Image.open(f"{self.data_dir}/img/{index}.jpg")
        sketch_img = Image.open(f"{self.data_dir}/sk/{index}.jpg")
        if self.transform:
            photo_img = self.transform(photo_img)
            sketch_img = self.transform(sketch_img)
            # sketch_photo_chunk = torch.stack([sketch_img, photo_img])
        return photo_img, random.randint(0,9)

def get_sketchy_pair(photo_path):
    sketch_path = (photo_path.rsplit('.', 1)[0] + '-1.' + photo_path.rsplit('.', 1)[1]).replace('photo', 'sketch').replace('jpg','png')
    return photo_path, sketch_path

# class CIFAR10NoClass(datasets.CIFAR10):
    # ds = fdatasets(root="./data", train=True, download=True, transform=transform)  
def get_ds(config, ds_name):
    if ds_name == 'cifar':
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((32,32)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        # channels = 3
        model = DiT_Llama(
            config.data.num_channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()
        ds = fdatasets(root="./data", train=True, download=True, transform=transform)
        
    elif ds_name == 'mnist':
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        ds = fdatasets(root="./data", train=True, download=True, transform=transform)
    elif  ds_name == 'sketchy':
        fdatasets = SketchyDataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        ds = fdatasets(transform=transform)
    elif ds_name == 'sketchy32':
        fdatasets = Sketchy32Dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # channels = 3
        ds = fdatasets(transform=transform)
    elif ds_name == 'sketchy32nocond':
        fdatasets = Sketchy32NocondDataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # channels = 3
        ds = fdatasets(transform=transform)
    else:
        raise NotImplementedError("This dataset is not yet implemented.")
    
    return ds, transform
        