import argparse
import os
import uuid

import cv2
import gradio as gr
import kornia
import numpy as np
import torch
from loguru import logger
from torchaudio.io import StreamReader
from torchaudio.io import StreamWriter

from benchmark.face_pipeline import alignFace
from benchmark.face_pipeline import FaceDetector
from benchmark.face_pipeline import inverse_transform_batch
from benchmark.face_pipeline import SoftErosion
from configs.train_config import TrainConfig
from models.model import HifiFace


class VideoSwap:
    def __init__(self, cfg, model=None):
        self.facedetector = FaceDetector(cfg.face_detector_weights)
        self.alignface = alignFace()
        self.work_dir = "."
        opt = TrainConfig()
        opt.use_ddp = False
        self.device = "cuda"
        self.ffmpeg_device = cfg.ffmpeg_device
        self.num_frames = 10
        self.kps_window = []
        checkpoint = (cfg.model_path, cfg.model_idx)
        if model is None:
            self.model = HifiFace(
                opt.identity_extractor_config, is_training=False, device=self.device, load_checkpoint=checkpoint
            )
        else:
            self.model = model
        self.model.eval()
        os.makedirs(self.work_dir, exist_ok=True)
        uid = uuid.uuid4()
        self.swapped_video = os.path.join(self.work_dir, f"tmp_{uid}.mp4")

        # model-idx_image-name_target-video-name.mp4
        swapped_with_audio_name = f"result_{uid}.mp4"

        # 带有音频的换脸视频
        self.swapped_video_with_audio = os.path.join(self.work_dir, swapped_with_audio_name)

        self.smooth_mask = SoftErosion(kernel_size=7, threshold=0.9, iterations=7).to(self.device)

    def yuv_to_rgb(self, img):
        img = img.to(torch.float)
        y = img[..., 0, :, :]
        u = img[..., 1, :, :]
        v = img[..., 2, :, :]
        y /= 255

        u = u / 255 - 0.5
        v = v / 255 - 0.5

        r = y + 1.14 * v
        g = y + -0.396 * u - 0.581 * v
        b = y + 2.029 * u

        rgb = torch.stack([r, g, b], -1)
        return rgb

    def rgb_to_yuv(self, img):
        r = img[..., 0, :, :]
        g = img[..., 1, :, :]
        b = img[..., 2, :, :]
        y = (0.299 * r + 0.587 * g + 0.114 * b) * 255
        u = (-0.1471 * r - 0.2889 * g + 0.4360 * b) * 255 + 128
        v = (0.6149 * r - 0.5149 * g - 0.1 * b) * 255 + 128
        yuv = torch.stack([y, u, v], -1)
        return torch.clamp(yuv, 0.0, 255.0, out=None).type(dtype=torch.uint8).transpose(3, 2).transpose(2, 1)

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

    def smooth_kps(self, kps):
        self.kps_window.append(kps.flatten())
        self.kps_window = self.kps_window[1:]
        X = np.stack(self.kps_window, axis=1)
        y = self.kps_window[-1]
        y_cor = X @ np.linalg.inv(X.transpose() @ X - 0.0007 * np.eye(self.num_frames)) @ X.transpose() @ y
        self.kps_window[-1] = y_cor
        return y_cor.reshape((5, 2))

    def detect_and_align(self, image, src_is=False):
        detection = self.facedetector(image)
        if detection.score is None:
            self.kps_window = []
            return None, None
        max_score_ind = np.argmax(detection.score, axis=0)
        kps = detection.key_points[max_score_ind]
        if len(self.kps_window) < self.num_frames:
            self.kps_window.append(kps.flatten())
        else:
            kps = self.smooth_kps(kps)
        align_img, warp_mat = self.alignface.align_face(image, kps, 256)
        align_img = cv2.resize(align_img, (256, 256))
        align_img = align_img.transpose(2, 0, 1)
        align_img = torch.from_numpy(align_img).unsqueeze(0).to(self.device).float()
        align_img = align_img / 255.0
        if src_is:
            self.kps_window = []
        return align_img, warp_mat

    def inference(self, source_face, target_video, shape_rate, id_rate, iterations=1):
        video = cv2.VideoCapture(target_video)
        # 获取视频宽度
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频高度
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧率
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        video.release()
        self.frame_size = (frame_height, frame_width)
        if self.ffmpeg_device == "cuda":
            self.decode_config = {"frames_per_chunk": 1, "decoder": "h264", "format": "yuv444p"}
            # self.decode_config = {
            #     "frames_per_chunk": 1,
            #     "decoder": "h264_cuvid",
            #     "decoder_option": {"gpu": "0"},
            #     "hw_accel": "cuda:0",
            # }

            self.encode_config = {
                "encoder": "h264_nvenc",  # GPU Encoder
                "encoder_format": "yuv444p",
                "encoder_option": {"gpu": "0", "cq": "10"},  # Run encoding on the cuda:0 device
                "hw_accel": "cuda:0",  # Data comes from cuda:0 device
                "frame_rate": frame_rate,
                "height": frame_height,
                "width": frame_width,
                "format": "yuv444p",
            }
        else:
            self.decode_config = {"frames_per_chunk": 1, "decoder": "h264", "format": "yuv444p"}

            self.encode_config = {
                "encoder": "libx264",
                "encoder_format": "yuv444p",
                "frame_rate": frame_rate,
                "height": frame_height,
                "width": frame_width,
                "format": "yuv444p",
            }
        src = source_face
        src, _ = self.detect_and_align(src, src_is=True)
        logger.info("start swapping")
        sr = StreamReader(target_video)
        if self.ffmpeg_device == "cpu":
            sr.add_basic_video_stream(**self.decode_config)
        else:
            sr.add_video_stream(**self.decode_config)
        sw = StreamWriter(self.swapped_video)
        sw.add_video_stream(**self.encode_config)
        with sw.open():
            for (chunk,) in sr.stream():
                # StreamReader cuda decode颜色格式默认为yuv需要转为rgb
                chunk = self.yuv_to_rgb(chunk)
                image = (chunk * 255).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                align_img, warp_mat = self.detect_and_align(image)
                chunk = chunk.transpose(3, 2).transpose(2, 1).to(self.device)
                if align_img is None:
                    result_face = chunk
                else:
                    with torch.no_grad():
                        for _ in range(iterations):
                            swapped_face, m_r = self.model.forward(src, align_img, shape_rate, id_rate)
                            swapped_face = torch.clamp(swapped_face, 0, 1)
                            align_img = swapped_face
                        smooth_face_mask, _ = self.smooth_mask(m_r)
                    warp_mat = torch.from_numpy(warp_mat).float().unsqueeze(0)
                    inverse_warp_mat = inverse_transform_batch(warp_mat)
                    swapped_face, smooth_face_mask = self._geometry_transfrom_warp_affine(
                        swapped_face, inverse_warp_mat, self.frame_size, smooth_face_mask
                    )
                    result_face = (1 - smooth_face_mask) * chunk + smooth_face_mask * swapped_face
                result_face = self.rgb_to_yuv(result_face).to(self.ffmpeg_device)
                sw.write_video_chunk(0, result_face)

        # 将target_video中的音频转移到换脸视频上
        command = f"ffmpeg -loglevel error -i {self.swapped_video} -i {target_video} -c copy \
            -map 0 -map 1:1? -y -shortest {self.swapped_video_with_audio}"
        os.system(command)

        # 删除没有音频的换脸视频
        os.system(f"rm {self.swapped_video}")
        return self.swapped_video_with_audio


class ConfigPath:
    face_detector_weights = "/mnt/c/yangguo/useful_ckpt/face_detector/face_detector_scrfd_10g_bnkps.onnx"
    model_path = ""
    model_idx = 80000
    ffmpeg_device = "cuda"


def main():
    cfg = ConfigPath()
    parser = argparse.ArgumentParser(
        prog="benchmark", description="What the program does", epilog="Text at the bottom of help"
    )
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-i", "--model_idx")
    parser.add_argument("-f", "--ffmpeg_device")

    args = parser.parse_args()

    cfg.model_path = args.model_path
    cfg.model_idx = int(args.model_idx)
    cfg.ffmpeg_device = args.ffmpeg_device

    infer = VideoSwap(cfg)

    def inference(source_face, target_video, shape_rate, id_rate):
        return infer.inference(source_face, target_video, shape_rate, id_rate)

    output = gr.Video(value=None, label="换脸结果")
    demo = gr.Interface(
        fn=inference,
        inputs=[
            gr.Image(shape=None, label="选脸图"),
            gr.Video(value=None, label="目标视频"),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="3d结构相似度（1.0表示完全替换）",
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="人脸特征相似度（1.0表示完全替换）",
            ),
        ],
        outputs=output,
        title="HiConFace视频人脸融合系统",
        description="v1.0: developed by yiwise CV group",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
