import argparse
import os

import cv2
import kornia
import numpy as np
import torch
from loguru import logger

from benchmark.face_pipeline import alignFace
from benchmark.face_pipeline import FaceDetector
from benchmark.face_pipeline import inverse_transform_batch
from benchmark.face_pipeline import SoftErosion
from configs.train_config import TrainConfig
from models.model import HifiFace


class ImageSwap:
    def __init__(self, cfg):
        self.source_face = cfg.source_face
        self.target_face = cfg.target_face
        self.facedetector = FaceDetector(cfg.face_detector_weights)
        self.alignface = alignFace()
        self.work_dir = cfg.work_dir
        opt = TrainConfig()
        opt.use_ddp = False
        self.device = "cuda"
        checkpoint = (cfg.model_path, cfg.model_idx)
        self.model = HifiFace(
            opt.identity_extractor_config, is_training=False, device=self.device, load_checkpoint=checkpoint
        )
        self.model.eval()
        os.makedirs(self.work_dir, exist_ok=True)

        # model-idx_swapped_src-image-name_target-face-name.jpg
        swapped_image_name = (
            str(cfg.model_idx)
            + "_"
            + "swapped"
            + "_"
            + os.path.basename(self.source_face).split(".")[0]
            + "_"
            + os.path.basename(self.target_face).split(".")[0]
            + ".jpg"
        )
        self.swapped_image = os.path.join(self.work_dir, swapped_image_name)
        self.smooth_mask = SoftErosion(kernel_size=7, threshold=0.9, iterations=7).to(self.device)

    def _geometry_transfrom_warp_affine(self, swapped_image, inv_att_transforms, frame_size, square_mask):
        swapped_image = kornia.geometry.transform.warp_affine(
            swapped_image,
            inv_att_transforms,
            frame_size,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
            fill_value=torch.zeros(3),
        )

        square_mask = kornia.geometry.transform.warp_affine(
            square_mask,
            inv_att_transforms,
            frame_size,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
            fill_value=torch.zeros(3),
        )
        return swapped_image, square_mask

    def detect_and_align(self, image):
        detection = self.facedetector(image)
        if detection.score is None:
            self.kps_window = []
            return None, None
        max_score_ind = np.argmax(detection.score, axis=0)
        kps = detection.key_points[max_score_ind]
        align_img, warp_mat = self.alignface.align_face(image, kps, 256)
        align_img = cv2.resize(align_img, (256, 256))
        align_img = align_img.transpose(2, 0, 1)
        align_img = torch.from_numpy(align_img).unsqueeze(0).to(self.device).float()
        align_img = align_img / 255.0
        return align_img, warp_mat

    def inference(self):
        src = cv2.cvtColor(cv2.imread(self.source_face), cv2.COLOR_BGR2RGB)
        src, _ = self.detect_and_align(src)
        if src is None:
            print("no face in src_img")
            return
        target = cv2.cvtColor(cv2.imread(self.target_face), cv2.COLOR_BGR2RGB)
        align_target, warp_mat = self.detect_and_align(target)
        if align_target is None:
            print("no face in target_img")
            return
        logger.info("start swapping")
        frame_size = (target.shape[0], target.shape[1])
        with torch.no_grad():
            swapped_face, m_r = self.model.forward(src, align_target)
            swapped_face = torch.clamp(swapped_face, 0, 1)
            smooth_face_mask, _ = self.smooth_mask(m_r)
        warp_mat = torch.from_numpy(warp_mat).float().unsqueeze(0)
        inverse_warp_mat = inverse_transform_batch(warp_mat)
        swapped_face, smooth_face_mask = self._geometry_transfrom_warp_affine(
            swapped_face, inverse_warp_mat, frame_size, smooth_face_mask
        )
        target = torch.from_numpy(target.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float() / 255.0
        result_face = (1 - smooth_face_mask) * target + smooth_face_mask * swapped_face
        result_face = torch.clamp(result_face * 255.0, 0.0, 255.0, out=None).type(dtype=torch.uint8)
        result_face = result_face.detach().cpu().numpy()
        img = result_face.transpose(0, 2, 3, 1)[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.swapped_image, img)


class ConfigPath:
    source_face = ""
    target_face = ""
    work_dir = ""
    face_detector_weights = "/mnt/c/yangguo/useful_ckpt/face_detector/face_detector_scrfd_10g_bnkps.onnx"
    model_path = ""
    model_idx = 80000


def main():
    cfg = ConfigPath()
    parser = argparse.ArgumentParser(
        prog="benchmark", description="What the program does", epilog="Text at the bottom of help"
    )
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-i", "--model_idx")
    parser.add_argument("-s", "--source_face")
    parser.add_argument("-t", "--target_face")
    parser.add_argument("-w", "--work_dir")

    args = parser.parse_args()
    cfg.source_face = args.source_face
    cfg.target_face = args.target_face
    cfg.model_path = args.model_path
    cfg.model_idx = int(args.model_idx)
    cfg.work_dir = args.work_dir
    infer = ImageSwap(cfg)
    infer.inference()


if __name__ == "__main__":
    main()
