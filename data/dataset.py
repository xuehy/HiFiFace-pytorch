import os
import random
from pathlib import Path
from typing import Dict
from typing import List

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.mode import FaceSwapMode
from configs.train_config import TrainConfig


class ManyToManyTrainDataset(Dataset):
    def __init__(self, img_root: str, mask_root: str, same_rate=0.5):
        """
        Many-to-many 训练数据集构建
        Parameters:
        -----------
        img_root: str, 人脸图片的根目录
        mask_root: str, 人脸图片mask的根目录
        same_rate: float, 每个batch里面相同人脸所占的比例
        """
        super(ManyToManyTrainDataset, self).__init__()
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

        self.img_files: List[str] = list(self.img_files)
        self.img_files.sort()

        self.mask_files: List[str] = list(self.mask_files)
        self.mask_files.sort()

        # 把img_files list按照id整理
        self.img_per_id: Dict[str, List[str]] = {}
        for idx, img_path in enumerate(self.img_files):
            id_name = img_path.split("/")[-2]
            if id_name in self.img_per_id.keys():
                self.img_per_id[id_name].append(idx)
            else:
                self.img_per_id[id_name] = [idx]

        self.id_list: List[str] = list(self.img_per_id.keys())

        # 所有id都遍历一遍，视为一个epoch
        self.length = len(self.id_list)

        logger.info(f"dataset contains {self.length} ids and {len(self.mask_files)} images")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        source_id_index = index
        source_index = random.choice(self.img_per_id[self.id_list[source_id_index]])
        if random.random() < self.same_rate:
            # 在相同id的文件列表中选择
            target_index = random.choice(self.img_per_id[self.id_list[source_id_index]])
            same = torch.ones(1)
        else:
            # 在不同id的文件列表中选择
            target_id_index = random.choice(list(set(range(self.length)) - set([source_id_index])))
            target_index = random.choice(self.img_per_id[self.id_list[target_id_index]])
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
            # "source_img_name": self.img_files[source_index],
            # "target_img_name": self.img_files[target_index],
            # "target_mask_name": self.mask_files[target_index],
        }


class ManyToOneTrainDataset(Dataset):
    def __init__(self, img_root: str, mask_root: str, target_name: str, same_rate=0.5):
        """
        Many-to-one 训练数据集构建
        Parameters:
        -----------
        img_root: str, 人脸图片的根目录
        mask_root: str, 人脸图片mask的根目录
        target_name: str, 目标脸id的名称, many-to-one里面的one
        same_rate: float, 每个batch里面相同人脸所占的比例
        """
        super(ManyToOneTrainDataset, self).__init__()
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
        self.target_name = target_name

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

        self.img_files: List[str] = list(self.img_files)
        self.img_files.sort()

        self.mask_files: List[str] = list(self.mask_files)
        self.mask_files.sort()

        # 把img_files list按照id整理
        self.img_per_id: Dict[str, List[str]] = {}
        for idx, img_path in enumerate(self.img_files):
            id_name = img_path.split("/")[-2]
            if id_name in self.img_per_id.keys():
                self.img_per_id[id_name].append(idx)
            else:
                self.img_per_id[id_name] = [idx]

        self.id_list: List[str] = list(self.img_per_id.keys())

        try:
            self.target_id_index: int = self.id_list.index(self.target_name)
        except Exception:
            raise Exception(f"{self.target_name} not in dataset dir")

        # 所有id都遍历一遍，视为一个epoch
        self.length = len(self.id_list)

        logger.info(f"dataset contains {self.length} ids and {len(self.mask_files)} images")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        source_id_index = index
        source_index = random.choice(self.img_per_id[self.id_list[source_id_index]])
        if random.random() < self.same_rate:
            # 在相同id的文件列表中选择
            target_index = random.choice(self.img_per_id[self.id_list[source_id_index]])
            same = torch.ones(1)
        else:
            # 直接选择target name中的图片
            target_index = random.choice(self.img_per_id[self.target_name])
            # 如果和source同个id
            if source_id_index == self.target_id_index:
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
            # "source_img_name": self.img_files[source_index],
            # "target_img_name": self.img_files[target_index],
            # "target_mask_name": self.mask_files[target_index],
        }


class TrainDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self):
        """Initialize this class"""
        opt = TrainConfig()
        if opt.mode is FaceSwapMode.MANY_TO_MANY:
            self.dataset = ManyToManyTrainDataset(opt.img_root, opt.mask_root, opt.same_rate)
        elif opt.mode is FaceSwapMode.MANY_TO_ONE:
            logger.info(f"In many-to-one mode, target face is {opt.target_name}")
            self.dataset = ManyToOneTrainDataset(opt.img_root, opt.mask_root, opt.target_name, opt.same_rate)
        else:
            raise NotImplementedError
        logger.info(f"dataset {type(self.dataset).__name__} created")
        if opt.use_ddp:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                num_workers=int(opt.num_threads),
                drop_last=True,
                sampler=self.train_sampler,
                pin_memory=True,
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads),
                drop_last=True,
                pin_memory=True,
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


if __name__ == "__main__":
    dataloader = TrainDatasetDataLoader()
    for idx, data in enumerate(dataloader):
        print(data["source_img_name"])
        print(data["target_img_name"])
        print(data["target_mask_name"])
        print(data["same"])
