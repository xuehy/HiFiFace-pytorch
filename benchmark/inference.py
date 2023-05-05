import argparse
import os
import uuid

import cv2
import kornia
import numpy as np
import torch
from loguru import logger

from benchmark.face_pipeline import alignFace
from benchmark.face_pipeline import FaceDetector
from benchmark.face_pipeline import inverse_transform_batch
from benchmark.face_pipeline import tensor2img
from configs.train_config import TrainConfig
from models.model import HifiFace
from torchaudio.io import StreamReader, StreamWriter


class VideoSwap:
    def __init__(self, cfg):
        self.source_face = cfg.source_face
        self.target_video = cfg.target_video
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
        self.tmp_dir = os.path.join(self.work_dir, str(uuid.uuid4()))
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.swapped_video = os.path.join(self.tmp_dir, "swapped_video.mp4")
        self.FFMPEG_COMMAND = "/data/tools/ffmpeg-5.1.1-amd64-static/ffmpeg"
        
        video = cv2.VideoCapture(self.target_video)
        # 获取视频宽度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频高度
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧率
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        video.release()
        
        self.encode_config = {
            "encoder": "h264_nvenc",  # GPU Encoder
            "encoder_format": "rgb0",
            "encoder_option": {"gpu": "0"},  # Run encoding on the cuda:0 device
            "hw_accel": "cuda:0",  # Data comes from cuda:0 device
            "frame_rate": frame_rate,
            "height": frame_height,
            "width": frame_width,
            "format": "rgb24",
        }

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
        logger.info("start swapping")
        
        sr = StreamReader(self.target_video)
        sr.add_basic_video_stream(1,format='rgb24')
        sw = StreamWriter(self.swapped_video)
        sw.add_video_stream(**self.encode_config)
        with sw.open():
            for (chunk, ) in sr.stream():
                image = chunk[0].numpy().transpose(1,2,0)
                align_img, warp_mat = self.detect_and_align(image)
                frame_size = (chunk.shape[2], chunk.shape[3])
                with torch.no_grad():
                    swapped_face = self.model.forward(src, align_img)
                    swapped_face = torch.clamp(swapped_face, 0, 1)
                chunk = (chunk.float()/255.0).cuda()
                warp_mat = torch.from_numpy(warp_mat).float().unsqueeze(0)
                inverse_warp_mat = inverse_transform_batch(warp_mat)
                square_mask = torch.ones_like(swapped_face).cuda()
                swapped_face, square_mask = self._geometry_transfrom_warp_affine(
                    swapped_face, inverse_warp_mat, frame_size, square_mask
                )
                result_face = (1 - square_mask) * chunk + square_mask * swapped_face
                result_face = torch.clamp(result_face*255.0, 0.0, 255.0, out=None).type(dtype=torch.uint8)
                sw.write_video_chunk(0, result_face)


class ConfigPath:
    source_face = ""
    target_video = ""
    work_dir = ""
    face_detector_weights = "/data/useful_ckpt/face_detector/face_detector_scrfd_10g_bnkps.onnx"
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
    parser.add_argument("-t", "--target_video")
    parser.add_argument("-w", "--work_dir")

    args = parser.parse_args()
    cfg.source_face = args.source_face
    cfg.target_video = args.target_video
    cfg.model_path = args.model_path
    cfg.model_idx = int(args.model_idx)
    cfg.work_dir = args.work_dir
    infer = VideoSwap(cfg)
    infer.inference()


if __name__ == "__main__":
    main()
