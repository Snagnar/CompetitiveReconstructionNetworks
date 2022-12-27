from xml.dom import minidom
import random
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from imageio import imread
from PIL import Image
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10, MNIST
from .datasets import get_cifar_anomaly_dataset, get_mnist_anomaly_dataset
from utils.utils import check_paths

def random_obscure(images):
    obscure_transform = transforms.Compose((
        transforms.RandomErasing(),
        transforms.RandomInvert(),
        transforms.RandomSolarize(0.0),
        transforms.RandomApply([transforms.ColorJitter()], 0.5)
    ))
    obscured = images.clone()
    for idx, image in enumerate(obscured):
        if idx % 2 == 0:
            obscured[idx] = obscure_transform(image)
    return obscured, torch.Tensor([int(not torch.equal(image, obscure)) for (image, obscure) in zip(images, obscured)])
        
        
        

class Cifar10AnomalyDataset(Dataset):
    
    def __init__(self, data_dir, train=True, imsize=32, abnormal_class_name=0):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")
        
        transform = transforms.Compose([transforms.Resize(imsize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_dataset, test_dataset = get_cifar_anomaly_dataset(train_dataset, test_dataset, abnormal_class_name)
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return self.dataset[index][0]
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
class MNISTAnomalyDataset(Dataset):
    
    def __init__(self, data_dir, train=True, abnormal_class_idx=0, imsize=128):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")

        transform = transforms.Compose([transforms.Resize(imsize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])


        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        train_dataset, test_dataset = get_mnist_anomaly_dataset(train_dataset, test_dataset, int(abnormal_class_idx))
        if train:
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return self.dataset[index][0]
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class PanoramaDataset(Dataset):
    
    def __init__(self, data_dir, inference=False, train=True, cache_images=True, train_split=0.7):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.train = train
        self.cache_images = cache_images
        self.inference = inference
        self.image_metadata_files = [p_file for p_file in self.data_dir.iterdir() if p_file.suffix == ".xml"]
        self.image_metadata_files.sort()
        self.good_ones = []
        self.bad_ones = []
        for metadata_file in self.image_metadata_files:
            metadata = minidom.parse(metadata_file.open())
            objects = metadata.getElementsByTagName("object")
            if len(objects) <= 1:
                self.good_ones.append(metadata_file.parent / (metadata_file.stem + ".jpg"))
            else:
                self.bad_ones.append(metadata_file.parent / (metadata_file.stem + ".jpg"))
        print("num good panorama images:", len(self.good_ones), "bad ones:", len(self.bad_ones))
        split_idx = int(len(self.good_ones) * train_split)
        if self.train:
            # self.image_files = self.good_ones[:int(len(self.good_ones) * train_split)]
            self.dataset = [(image_file, 0) for image_file in self.good_ones[:split_idx]]
        else:
            self.dataset = [(image_file, 1) for image_file in self.bad_ones[:min(2 * split_idx, len(self.bad_ones))]]
            self.dataset += [(image_file, 0) for image_file in self.good_ones[split_idx:]]
            
            # for directory in (self.data_dir / "test").iterdir():
            #     self.image_files += list(directory.iterdir())
        print(self.train, len(self.dataset))
        random.shuffle(self.dataset)
        self.images = None
        if cache_images:
            self.images = []
            logging.info("caching images...")
            for image_file, _ in self.dataset:
                image = Image.open(image_file).convert("RGB")
                image = self.transform(image)
                self.images.append(image)

    def __getitem__(self, index):
        image_file, target = self.dataset[index]
        if self.cache_images and self.images[index] is not None:
            image = self.images[index]
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
        
        if self.train:
            return image
        else:
            # if image_file.parent.name == "good":
            #     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            # else:
            #     target = Image.open(
            #         image_file.replace("/test/", "/ground_truth/").replace(
            #             ".png", "_mask.png"
            #         )
            #     )
            #     target = self.target_transform(target)
            if self.inference:
                return image, image_file.stem
            else:
                return image, target

    def __len__(self):
        return len(self.dataset)

class MVTecDataset(Dataset):

    def __init__(self, data_dir, inference=False, train=True, include_random_images=False, cache_images=True, imsize=128):
        self.data_dir = Path(data_dir)
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"direcotry {str(self.data_dir.resolve())} is not a directory!")
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train = train
        self.cache_images = cache_images
        self.inference = inference
        if self.train:
            self.image_files = list((self.data_dir / "train" / "good").iterdir())
        else:
            self.image_files = []
            for directory in (self.data_dir / "test").iterdir():
                self.image_files += list(directory.iterdir())
            if include_random_images:
                other_cats = list(self.data_dir.parent.iterdir())
                random_files = []
                selected = other_cats[random.randint(0, len(other_cats) - 1)]
                print("got selected:", str(selected), other_cats, random.randint(0, len(other_cats) - 1))
                for directory in (selected / "test").iterdir():
                    if directory.name != "good":
                        random_files += list(directory.iterdir())
                for _ in range(40):
                    self.image_files.append(random_files[random.randint(0, len(random_files) - 1)])
                train_image_files = list((self.data_dir / "train" / "good").iterdir())
                for _ in range(40):
                    self.image_files.append(train_image_files[random.randint(0, len(train_image_files) - 1)])
                
        self.image_files = sorted(self.image_files)
        self.image_files = [image for image in self.image_files if image.suffix == ".png"]
        random.shuffle(self.image_files)
        self.images = None
        if cache_images:
            self.images = []
            logging.info("caching images...")
            for image_file in self.image_files:
                image = Image.open(image_file).convert("RGB")
                image = self.transform(image)
                self.images.append(image)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        if self.cache_images and self.images[index] is not None:
            image = self.images[index]
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
        
        if self.train:
            return image
        else:
            # if image_file.parent.name == "good":
            #     target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            # else:
            #     target = Image.open(
            #         image_file.replace("/test/", "/ground_truth/").replace(
            #             ".png", "_mask.png"
            #         )
            #     )
            #     target = self.target_transform(target)
            if self.inference:
                return image, image_file.stem
            else:
                target = int(image_file.parent.name != "good")
                return image, target

    def __len__(self):
        return len(self.image_files)

class RoadImageDataset(Dataset):
    def __init__(self, data_dir, train=True, cache=True, inference=False, imsize=128):
        self.data_dir = Path(data_dir)
        if not train:
            logging.warn("No testing partition available yet for road image dataset.")
            return

        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"directory {str(self.data_dir.reslove())} is not a directory!")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imsize, imsize)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        intensity_dir = self.data_dir / "intensity_files"
        height_dir = self.data_dir / "height_files"
        self.inference = inference
        self.files = list(zip(sorted(list(height_dir.iterdir())), sorted(list(intensity_dir.iterdir()))))
        self.orig_image_size = None
        if len(self.files) == 0:
            raise RuntimeError(f"directory {str(self.data_dir.resolve())} does not contain any files!")
        self.cache = cache
        if cache:
            print("caching images...")
            self.dataset = [self.load_image(i) for i in range(len(self.files))]
    
    def load_image(self, idx):
        image_files = self.files[idx]
        intensity_values = imread(image_files[1])[:, :, np.newaxis]
        height_values = imread(image_files[0])[:, :, np.newaxis]
        if len(height_values.shape) > 3:
            height_values = height_values[:, :, 0, 0]
            height_values = height_values[:, :, np.newaxis]
            intensity_values = intensity_values[:, :, 0, 0]
            intensity_values = intensity_values[:, :, np.newaxis]
        image_values = np.concatenate([height_values, intensity_values, intensity_values], axis=-1)
        self.orig_image_size = image_values.shape
        image_values = self.transform(image_values).type(torch.FloatTensor)
        return (image_values, image_files[0].stem) if self.inference else image_values

    def __getitem__(self, idx):
        if self.cache:
            return self.dataset[idx]
        return self.load_image(idx)
    
    def __len__(self):
        return len(self.files)

class RoadImageDatasetPartition(Dataset):
    def __init__(self, image_files, train=False, imsize=128):
        super().__init__()
        self.train = train
        self.files = image_files
        print("in partition", "train" if train else "test", "files:", len(self.files))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((imsize, imsize)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __getitem__(self, idx):
        image_files = self.files[idx]
        intensity_values = imread(image_files[1])[:, :, np.newaxis]
        height_values = imread(image_files[0])[:, :, np.newaxis]
        if len(height_values.shape) > 3:
            height_values = height_values[:, :, 0, 0]
            height_values = height_values[:, :, np.newaxis]
            intensity_values = intensity_values[:, :, 0, 0]
            intensity_values = intensity_values[:, :, np.newaxis]
        image_values = np.concatenate([height_values, intensity_values, intensity_values], axis=-1)
        image_values = self.transform(image_values).type(torch.FloatTensor)
        return image_values if self.train else (image_values, int(image_files[0].parent.parent.name == "bad_ones"))
    
    def __len__(self):
        return len(self.files)

class AnnotatedRoadImageDataset(Dataset):
    train_split = 0.7
    def __init__(self, data_dir, train=True, inference=False, cache=True, imsize=128):
        super().__init__()
        # if train:
        self.data_dir = Path(data_dir) / "good_ones"
        check_paths(self.data_dir)
        if not self.data_dir.is_dir():
            raise RuntimeError(f"directory {str(self.data_dir.reslove())} is not a directory!")
        intensity_dir = self.data_dir / "intensity_files"
        height_dir = self.data_dir / "height_files"
        self.inference = inference                                                                                                                                                             
        self.files = list(zip(sorted(list(height_dir.iterdir())), sorted(list(intensity_dir.iterdir()))))
        if len(self.files) == 0:
            raise RuntimeError(f"directory {str(self.data_dir.resolve())} does not contain any files!")
        random.shuffle(self.files)
        # self.files = random.sample(self.files, len(self.files), )
        if train:
            self.dataset = RoadImageDatasetPartition(self.files[:int(self.train_split * len(self.files))], train=True, imsize=imsize)
        else:
            self.data_dir = self.data_dir.parent / "bad_ones"

            check_paths(self.data_dir)
            if not self.data_dir.is_dir():
                raise RuntimeError(f"directory {str(self.data_dir.reslove())} is not a directory!")
            intensity_dir = self.data_dir / "intensity_files"
            height_dir = self.data_dir / "height_files"
            self.inference = inference                                                                                                                                                             
            self.test_files = list(zip(sorted(list(height_dir.iterdir())), sorted(list(intensity_dir.iterdir()))))
            self.files = self.files[int(self.train_split * len(self.files)):] + self.test_files
            random.shuffle(self.files)
            self.dataset = RoadImageDatasetPartition(self.files, train=False, imsize=imsize)
        print(train, len(self.dataset))
        self.cache = cache
        self.cached_dataset = None
        if cache:
            print("caching...")
            self.cached_dataset = [self.dataset[idx] for idx in range(len(self.dataset))]
        
    
    def __getitem__(self, idx):
        if self.cache:
            return self.cached_dataset[idx]
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
