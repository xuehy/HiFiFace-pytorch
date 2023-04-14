import os
from typing import List
from typing import Optional

import cv2
import numpy as np
import torch

from configs.train_config import TrainConfig
from models.model import HifiFace


def test(
    data_root: str,
    result_path: str,
    source_face: List[str],
    target_face: List[str],
    model_path: str,
    model_idx: Optional[int],
):
    opt = TrainConfig()
    opt.use_ddp = False

    device = "cpu"
    checkpoint = (model_path, model_idx)
    model = HifiFace(opt.identity_extractor_config, is_training=False, device=device, load_checkpoint=checkpoint)
    model.eval()

    results = []
    for source, target in zip(source_face, target_face):
        source = os.path.join(data_root, source)
        target = os.path.join(data_root, target)

        src_img = cv2.imread(source)
        src_img = cv2.resize(src_img, (256, 256))
        src = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src = src.transpose(2, 0, 1)
        src = torch.from_numpy(src).unsqueeze(0).to(device).float()
        src = src / 255.0

        tgt_img = cv2.imread(target)
        tgt_img = cv2.resize(tgt_img, (256, 256))
        tgt = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
        tgt = tgt.transpose(2, 0, 1)
        tgt = torch.from_numpy(tgt).unsqueeze(0).to(device).float()
        tgt = tgt / 255.0

        with torch.no_grad():
            result_face = model.forward(src, tgt).cpu()
            result_face = torch.clamp(result_face, 0, 1) * 255
        result_face = result_face.numpy()[0].astype(np.uint8)
        result_face = result_face.transpose(1, 2, 0)

        result_face = cv2.cvtColor(result_face, cv2.COLOR_BGR2RGB)
        one_result = np.concatenate((src_img, tgt_img, result_face), axis=0)
        results.append(one_result)
    result = np.concatenate(results, axis=1)
    swapped_face = os.path.join(data_root, result_path)
    cv2.imwrite(swapped_face, result)


if __name__ == "__main__":
    data_root = "/home/xuehongyang/data/face_swap_test"

    model_path = "/data/checkpoints/hififace/baseline_1k_ddp_with_cyc_1681278017147"
    model_idx = 520000

    target = [
        "male_1.jpg",
        "male_2.jpg",
        "minlu_1.jpg",
        "minlu_2.jpg",
        "shizong_1.jpg",
        "shizong_2.jpg",
        "tianxin_1.jpg",
        "tianxin_2.jpg",
        "xiaohui_1.jpg",
        "xiaohui_2.jpg",
        "female_1.jpg",
        "female_2.jpg",
        "female_3.jpg",
        "female_4.jpg",
        "female_5.jpg",
        "female_6.jpg",
        "lixia_1.jpg",
        "lixia_2.jpg",
        "qq_1.jpg",
        "qq_2.jpg",
        "pink_1.jpg",
        "pink_2.jpg",
        "xulie_1.jpg",
        "xulie_2.jpg",
    ]

    source = ["gaoyuanyuan.jpg"] * len(target)
    target_src = os.path.join(data_root, "../result_1tom.jpg")
    test(data_root, target_src, source, target, model_path, model_idx)
