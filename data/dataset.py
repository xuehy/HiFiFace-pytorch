import os
import random
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.train_config import TrainConfig


class TrainDataset(Dataset):
    def __init__(self, img_root: str, mask_root: str, same_rate=0.5):
        """
        训练数据集构建
        Parameters:
        -----------
        img_root: str, 人脸图片的根目录
        mask_root: str, 人脸图片mask的根目录
        same_rate: float, 每个batch里面相同人脸所占的比例
        """
        super(TrainDataset, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.img_root = img_root
        self.mask_root = mask_root
        self.same_rate = same_rate

        self.mask_files = []

        img_dir = Path(img_root)
        self.img_files = set([str(x) for x in img_dir.glob(f"**/*.jpg")])

        mask_dir = Path(mask_root)
        self.mask_files = set([str(x) for x in mask_dir.glob(f"**/*.jpg")])

        # 移除没有对应mask的图片
        imgs_without_masks = []
        for img_path in self.img_files:
            base_img_path = "/".join(img_path.split("/")[-2:])
            mask_path = os.path.join(mask_root, base_img_path)
            if mask_path not in self.mask_files:
                imgs_without_masks.append(img_path)
        for img_path in imgs_without_masks:
            self.img_files.remove(img_path)

        self.img_files = list(self.img_files)
        self.img_files.sort()

        self.mask_files = list(self.mask_files)
        self.mask_files.sort()

        self.length = len(self.img_files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        length = self.__len__()
        source_index = index
        if random.random() < self.same_rate:
            target_index = source_index
        else:
            target_index = random.randrange(length)

        if target_index == source_index:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        target_img = Image.open(self.img_files[target_index]).convert("RGB")
        source_img = Image.open(self.img_files[source_index]).convert("RGB")

        target_mask = Image.open(self.mask_files[target_index]).convert("RGB")

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        target_mask = self.transform(target_mask)[0, :, :].unsqueeze(0)

        return {
            "source_image": source_img,
            "target_image": target_img,
            "target_mask": target_mask,
            "same": same,
        }


class TrainDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self):
        """Initialize this class"""
        opt = TrainConfig()
        self.dataset = TrainDataset(opt.img_root, opt.mask_root, opt.same_rate)
        logger.info(f"dataset {type(self.dataset).__name__} created")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.num_threads),
            drop_last=True,
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for data in self.dataloader:
            yield data
