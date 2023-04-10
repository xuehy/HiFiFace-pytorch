import os
from pathlib import Path

import cv2
import torch
from model import BiSeNet
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# For BiSeNet and for official_224 SimSwap


class MaskDataset(Dataset):
    def __init__(self, img_root, mask_root):
        img_dir = Path(img_root)
        self.to_tensor_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_files = list(img_dir.glob(f"**/*.jpg"))
        self.img_files.sort()
        self.mask_files = [os.path.join(mask_root, os.path.relpath(img_path, img_root)) for img_path in self.img_files]

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, index):
        img = Image.open(self.img_files[index]).convert("RGB")
        return {"img": self.to_tensor_normalize(img), "mask_path": self.mask_files[index]}


class MaskDataLoader:
    def __init__(self):
        """Initialize this class"""
        self.dataset = MaskDataset(img_root="/data/dataset/face_1k/alignHQ", mask_root="/data/dataset/face_1k/mask")

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=False
        )

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset) / 8

    def __iter__(self):
        """Return a batch of data"""
        for data in self.dataloader:
            yield data


if __name__ == "__main__":
    dataloader = MaskDataLoader()
    bisenet_path = "/data/useful_ckpt/face_parsing/parsing_model_79999_iter.pth"
    bisenet = BiSeNet(n_classes=19)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bisenet.to(device)
    state_dict = torch.load(bisenet_path, map_location=device)
    bisenet.load_state_dict(state_dict)
    bisenet.eval()

    for data in tqdm(dataloader):
        mask, ignore_ids = bisenet.get_mask(data["img"].to(device), 256)
        mask = (mask * 255).to(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1).repeat(3, 3)

        for i in range(mask.shape[0]):
            if ignore_ids[i]:
                continue
            path = data["mask_path"][i]
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            cv2.imwrite(path, mask[i])
