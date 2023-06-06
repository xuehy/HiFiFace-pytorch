import argparse

import gradio as gr

from benchmark.app_image import ImageSwap
from benchmark.app_video import VideoSwap
from configs.train_config import TrainConfig
from models.model import HifiFace


class ConfigPath:
    face_detector_weights = "/home/yangguo/useful_ckpt/face_detector_scrfd_10g_bnkps.onnx"
    model_path = ""
    model_idx = 80000
    ffmpeg_device = "cuda"
    device = "cuda"


def main():
    cfg = ConfigPath()
    parser = argparse.ArgumentParser(
        prog="benchmark", description="What the program does", epilog="Text at the bottom of help"
    )
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-i", "--model_idx")
    parser.add_argument("-f", "--ffmpeg_device")
    parser.add_argument("-d", "--device", default="cuda")

    args = parser.parse_args()

    cfg.model_path = args.model_path
    cfg.model_idx = int(args.model_idx)
    cfg.ffmpeg_device = args.ffmpeg_device
    cfg.device = args.device
    opt = TrainConfig()
    checkpoint = (cfg.model_path, cfg.model_id)
    model = HifiFace(opt.identity_extractor_config, is_training=False, device=cfg.device, load_checkpoint=checkpoint)

    image_infer = ImageSwap(cfg, model)
    video_infer = VideoSwap(cfg, model)

    def inference_image(source_face, target_face, shape_rate, id_rate):
        return image_infer.inference(source_face, target_face, shape_rate, id_rate)

    def inference_video(source_face, target_video, shape_rate, id_rate):
        return video_infer.inference(source_face, target_video, shape_rate, id_rate)

    with gr.Blocks(title="dd") as demo:
        gr.Markdown(
            """
        # HiConFace人脸融合系统
        v1.0: developed by yiwise CV group
        """
        )
        with gr.Tab("图片换脸"):

            output_image = gr.Image(shape=None, label="换脸结果")
            source_image = gr.Image(shape=None, label="选脸图")
            structure_sim = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="3d结构相似度（1.0表示最接近选脸图）",
            )
            id_sim = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="人脸特征相似度",
            )
            image_btn = gr.Button("图像换脸")
            image_btn.click(
                fn=inference_image,
                inputs=[
                    source_image,
                    gr.Image(shape=None, label="目标图"),
                    structure_sim,
                    id_sim,
                ],
                outputs=output_image,
            )

        with gr.Tab("视频换脸"):
            source_image = gr.Image(shape=None, label="选脸图")
            output_video = gr.Video(value=None, label="换脸结果")
            structure_sim = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="3d结构相似度（1.0表示最接近选脸图）",
            )
            id_sim = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="人脸特征相似度",
            )
            video_btn = gr.Button("视频换脸")
            video_btn.click(
                fn=inference_video,
                inputs=[
                    source_image,
                    gr.Video(value=None, label="目标视频"),
                    structure_sim,
                    id_sim,
                ],
                outputs=output_video,
            )

    demo.launch(server_name="0.0.0.0", server_port=7860)
