import os
import pickle
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
    def __init__(self, dataset_root: str, dataset_index: str, same_rate=0.5):
        """
        Many-to-many 训练数据集构建
        Parameters:
        -----------
        dataset_root: str, 数据集根目录
        dataset_index: str, 数据集index文件路径
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
        self.data_root = Path(dataset_root)
        with open(dataset_index, "rb") as f:
            self.file_index = pickle.load(f, encoding="bytes")

        self.same_rate = same_rate

        self.id_list: List[str] = list(self.file_index.keys())

        # 所有id都遍历一遍，视为一个epoch
        self.length = len(self.id_list)
        self.image_num = sum([len(v) for v in self.file_index.values()])

        self.mask_dir = "mask" if TrainConfig().mouth_mask else "mask_no_mouth"
        logger.info(f"dataset contains {self.length} ids and {self.image_num} images")
        logger.info(f"will use mask mode: {self.mask_dir}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        source_id_index = index
        source_file = random.choice(self.file_index[self.id_list[source_id_index]])
        if random.random() < self.same_rate:
            # 在相同id的文件列表中选择
            target_file = random.choice(self.file_index[self.id_list[source_id_index]])
            same = torch.ones(1)
        else:
            # 在不同id的文件列表中选择
            target_id_index = random.choice(list(set(range(self.length)) - set([source_id_index])))
            target_file = random.choice(self.file_index[self.id_list[target_id_index]])
            same = torch.zeros(1)

        source_file = self.data_root / Path(source_file)
        target_file = self.data_root / Path(target_file)
        target_mask_file = target_file.parent.parent.parent / self.mask_dir / target_file.parent.stem / target_file.name

        target_img = Image.open(target_file.as_posix()).convert("RGB")
        source_img = Image.open(source_file.as_posix()).convert("RGB")

        target_mask = Image.open(target_mask_file.as_posix()).convert("RGB")

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        target_mask = self.transform(target_mask)[0, :, :].unsqueeze(0)

        return {
            "source_image": source_img,
            "target_image": target_img,
            "target_mask": target_mask,
            "same": same,
            # "source_img_name": source_file.as_posix(),
            # "target_img_name": target_file.as_posix(),
            # "target_mask_name": target_mask_file.as_posix(),
        }


class OneToManyTrainDataset(Dataset):
    def __init__(self, dataset_root: str, dataset_index: str, source_name: str, same_rate=0.5):
        """
        One-to-many 训练数据集构建
        Parameters:
        -----------
        dataset_root: str, 数据集根目录
        dataset_index: str, 数据集index文件路径
        source_name: str, source face id的名称, one-to-many里面的one
        same_rate: float, 每个batch里面相同人脸所占的比例
        """
        super(OneToManyTrainDataset, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((256, 256)),
                transforms.ToTensor(),
            ]
        )
        self.data_root = Path(dataset_root)
        with open(dataset_index, "rb") as f:
            self.file_index = pickle.load(f, encoding="bytes")
        self.same_rate = same_rate
        self.source_name = source_name

        self.id_list: List[str] = list(self.file_index.keys())

        try:
            self.source_id_index: int = self.id_list.index(self.source_name)
        except Exception:
            raise Exception(f"{self.source_name} not in dataset dir")

        # 所有id都遍历一遍，视为一个epoch
        self.length = len(self.id_list)
        self.image_num = sum([len(v) for v in self.file_index.values()])
        self.mask_dir = "mask" if TrainConfig().mouth_mask else "mask_no_mouth"
        logger.info(f"dataset contains {self.length} ids and {self.image_num} images")
        logger.info(f"will use mask mode: {self.mask_dir}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        target_id_index = index
        target_file = random.choice(self.file_index[self.id_list[target_id_index]])
        if random.random() < self.same_rate:
            # 在相同id的文件列表中选择
            source_file = random.choice(self.file_index[self.id_list[target_id_index]])
            same = torch.ones(1)
        else:
            # 直接选择source name中的图片
            source_file = random.choice(self.file_index[self.source_name])
            # 如果和target同个id
            if self.source_id_index == target_id_index:
                same = torch.ones(1)
            else:
                same = torch.zeros(1)

        source_file = self.data_root / Path(source_file)
        target_file = self.data_root / Path(target_file)
        target_mask_file = target_file.parent.parent.parent / self.mask_dir / target_file.parent.stem / target_file.name

        target_img = Image.open(target_file.as_posix()).convert("RGB")
        source_img = Image.open(source_file.as_posix()).convert("RGB")

        target_mask = Image.open(target_mask_file.as_posix()).convert("RGB")

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        target_mask = self.transform(target_mask)[0, :, :].unsqueeze(0)

        return {
            "source_image": source_img,
            "target_image": target_img,
            "target_mask": target_mask,
            "same": same,
            # "source_img_name": source_file.as_posix(),
            # "target_img_name": target_file.as_posix(),
            # "target_mask_name": target_mask_file.as_posix(),
        }


class TrainDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self):
        """Initialize this class"""
        opt = TrainConfig()
        if opt.mode is FaceSwapMode.MANY_TO_MANY:
            self.dataset = ManyToManyTrainDataset(opt.dataset_root, opt.dataset_index, opt.same_rate)
        elif opt.mode is FaceSwapMode.ONE_TO_MANY:
            logger.info(f"In one-to-many mode, source face is {opt.source_name}")
            self.dataset = OneToManyTrainDataset(opt.dataset_root, opt.dataset_index, opt.source_name, opt.same_rate)
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
        # print(data["source_img_name"])
        # print(data["target_img_name"])
        # print(data["target_mask_name"])
        print(data["same"])
