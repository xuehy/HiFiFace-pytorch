from typing import Optional

import cv2
import numpy as np
import torch

from configs.train_config import TrainConfig
from models.model import HifiFace


def inference(source_face: str, target_video: str, model_path: str, model_idx: Optional[int], swapped_video: str):
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

    video = cv2.VideoCapture(target_video)

    target = cv2.VideoWriter(swapped_video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (256, 256), True)
    while True:
        still_reading, target_face = video.read()
        if not still_reading:
            break
        tgt = cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB)
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
        target.write(result_face)
    target.release()
    video.release()


if __name__ == "__main__":
    source_face = "/home/xuehongyang/data/face_swap_test/male_1.jpg"
    target_video = "/home/xuehongyang/data/face_swap_test/video_1.mp4"
    model_path = "/data/checkpoints/hififace/baseline_1k_ddp_with_cyc_1681278017147"
    model_idx = 520000
    swapped_face = "/home/xuehongyang/data/video_1_male_1.mp4"
    inference(source_face, target_video, model_path, model_idx, swapped_face)
