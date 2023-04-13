from typing import Optional

import cv2
import numpy as np
import torch

from configs.train_config import TrainConfig
from models.model import HifiFace


def inference(source_face: str, target_face: str, model_path: str, model_idx: Optional[int], swapped_face: str):
    opt = TrainConfig()
    opt.use_ddp = False

    device = "cpu"
    checkpoint = (model_path, model_idx)
    model = HifiFace(opt.identity_extractor_config, is_training=False, device=device, load_checkpoint=checkpoint)
    model.eval()

    src = cv2.cvtColor(cv2.imread(source_face), cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (256, 256))
    src = src.transpose(2, 0, 1)
    src = torch.from_numpy(src).unsqueeze(0).to(device).float()
    src = src / 255.0

    tgt = cv2.cvtColor(cv2.imread(target_face), cv2.COLOR_BGR2RGB)
    tgt = cv2.resize(tgt, (256, 256))
    tgt = tgt.transpose(2, 0, 1)
    tgt = torch.from_numpy(tgt).unsqueeze(0).to(device).float()
    tgt = tgt / 255.0

    with torch.no_grad():
        result_face = model.forward(src, tgt).cpu()
        result_face = torch.clamp(result_face, 0, 1) * 255
    result_face = result_face.numpy()[0].astype(np.uint8)
    result_face = result_face.transpose(1, 2, 0)

    result_face = cv2.cvtColor(result_face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(swapped_face, result_face)


if __name__ == "__main__":
    source_face = "/home/xuehongyang/data/female_1.jpg"
    target_face = "/home/xuehongyang/data/female_2.jpg"
    model_path = "/data/checkpoints/hififace/baseline_1k_ddp_with_cyc_1681278017147"
    model_idx = 80000
    swapped_face = "/home/xuehongyang/data/male_1_to_male_2.jpg"
    inference(source_face, target_face, model_path, model_idx, swapped_face)
