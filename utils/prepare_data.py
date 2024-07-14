import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import v2
import random
from torchvision import tv_tensors
import omegaconf
import numpy as np
from torch.utils.data import default_collate
import glob
import omegaconf
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, img_list:list[str], labels_dict: dict[str, int], cache_imgs: bool=False):
        # as the names are in order we can just use listdir
        self.img_list = img_list
        self.labels_dict = labels_dict
        self.chache_imgs = cache_imgs
        if self.chache_imgs:
            self.imgs = []
            self.labels = []
            print("Making chache of imgs")
            for i in tqdm(range(len(self.img_list))):
                img_path = self.img_list[i]
        
                img = ImageDataset.load_image(img_path, True)
                label = os.path.basename(os.path.dirname(img_path))
                
                label = self.labels_dict[label]
                self.imgs.append(img)
                self.labels.append(label)

                
        
    @staticmethod
    def load_image(img_path:str, resize:bool=False):
        img = Image.open(img_path)
        if resize:
            img = img.resize((224, 224))
        img_tensor = tv_tensors.Image(img) / 255.0
        
        return img_tensor
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        if not self.chache_imgs:
            img_path = self.img_list[idx]
            
            img = ImageDataset.load_image(img_path, True)
            label = os.path.basename(os.path.dirname(img_path))
            
            label = self.labels_dict[label]
        
        else:
            img = self.imgs[idx]
            label = self.labels[idx]
        
        return img, label
    
class ImageTransform:
    def __init__(self, mean_img:list[float], std_img:list[float], size: int=224) -> None:
        self.mean_img = mean_img
        self.std_img = std_img
        # self.size = size
        
    def transform(self, image, phase:str="train"):
        if phase=="train":
            # Random crop resize
            # top = random.randint(0, image.shape[-2]-self.size)
            # left = random.randint(0, image.shape[-2]-self.size)
            # image = v2.functional.resized_crop(image, top, left, self.size)
            
            # Random afine
            # rotation makes the validations loss diverge so set to 0
            degrees = 0
            translate = random.uniform(0, 0.15)
            # print(degrees, translate)
            image = v2.functional.affine(image, angle=degrees, translate=(translate, translate), scale=1.0, shear=0.0)

            # RandomHorizontalFlip 
            if random.random() > 0.5:
                image = v2.functional.horizontal_flip(image)

            # ElasticTransform
            displacement = v2.ElasticTransform(alpha=60.0)._get_params(image)['displacement']
            image = v2.functional.elastic(image, displacement)
            
            # adjust brightness and contrast
            brightness = random.uniform(0.92, 1.12)
            contrast = random.uniform(0.92, 1.12)
            image = v2.functional.adjust_brightness(image, brightness)
            image = v2.functional.adjust_contrast(image, contrast)
        # else:
        #     image = v2.functional.resize(image, self.size)
            
        # result of all the dataset
        image = v2.Normalize(mean=self.mean_img, std=self.std_img)(image)

        
        return image

        
    def __call__(self, img, phase:str="train"):
        return self.transform(img, phase)

# jpgのフォーマットしか使えない。
# リストにパスを格納
def make_datapath_list(base_dir):
    """
    データのパスを格納したリストを作成する
        
    Returns
    -------
    path_list : list
        データのパスを格納したリスト
    
    """
    target_path = os.path.join(base_dir+"/*.jpg")
    print(target_path)
    
    path_list = []
    #globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list
    
# フォルダの画像数枚を数える
def get_filenames(img_dir):
    image_filenames = os.listdir(img_dir)
    image_filenames = [os.path.join(img_dir, img) for img in image_filenames]
    
    return image_filenames

# 訓練と検証を分割する
def split_train_val(base_dir, train_ration=0.9):
    
    train_paths = []
    val_paths = []
    # 最初はフォルダの名前を格納する
    folders_list = get_filenames(base_dir)
    
    for folder in folders_list:
        paths_list = make_datapath_list(folder)
        
        random.shuffle(paths_list)
        
        # 分割比率を使って
        total_samples = len(paths_list)
        train_samples = int(train_ration * total_samples)
        
        # リストを訓練データと検証データに分割
        train_data = paths_list[:train_samples]
        val_data = paths_list[train_samples:]
        
        train_paths.extend(train_data)
        val_paths.extend(val_data)
    
    return train_paths, val_paths



def collate_fn(
    batch: list[np.ndarray, int],
    device: torch.device,
    transform: callable=None,
    phase: str="train") -> tuple[torch.Tensor]:
    
    batch = default_collate(batch)
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)
    if transform is not None:
        batch[0] = transform(batch[0], phase)

    return batch

def create_dataset(conf: omegaconf.DictConfig, cache_imgs: bool=True):
    base_dir = r"".join(conf.train.base_dir)
    train_list, val_list = split_train_val(base_dir)
    
    # with this we get the images between -1 and 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    pilots_labels = {"ayanami_rei": 0, "ikari_shinji": 1, "makinami_mari_illustrious": 2,
                "nagisa_kaworu": 3, "souryuu_asuka_langley": 4}

    train_data = ImageDataset(train_list, pilots_labels, cache_imgs=cache_imgs)
    val_data = ImageDataset(val_list, pilots_labels, cache_imgs=cache_imgs)
    transforms = ImageTransform(mean_img = mean, std_img = std)
    
    train_loader = DataLoader(train_data, batch_size=conf.train.batch_size, collate_fn=lambda batch: collate_fn(batch, conf.train.device, transforms, "train"), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=conf.train.eval_batch_size, collate_fn=lambda batch: collate_fn(batch, conf.train.device, transforms, "val"), shuffle=False)

    return train_loader, val_loader

def prepare_test(conf: omegaconf.DictConfig):
    base_dir = r"".join(conf.train.base_dir)
    train_list, val_list = split_train_val(base_dir)
    
        # with this we get the images between -1 and 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    pilots_labels = {"ayanami_rei": 0, "ikari_shinji": 1, "makinami_mari_illustrious": 2,
                "nagisa_kaworu": 3, "souryuu_asuka_langley": 4}

    val_data = ImageDataset(val_list, pilots_labels, cache_imgs=False)
    transforms = ImageTransform(mean_img = mean, std_img = std)
    
    val_loader = DataLoader(val_data, batch_size=16, collate_fn=lambda batch: collate_fn(batch, conf.train.device, transforms, "val"), shuffle=False)

    return val_loader