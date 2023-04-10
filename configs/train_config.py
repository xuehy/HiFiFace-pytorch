import os
import time
from dataclasses import dataclass

from configs.singleton import Singleton


@Singleton
@dataclass
class TrainConfig:
    img_root: str = "/data/dataset/face_small/"
    mask_root: str = "/data/dataset/face_mask_small"
    batch_size: int = 12
    num_threads: int = 24
    same_rate: float = 0.2
    lr: float = 2e-5
    grad_clip: float = 0.1
    amp: bool = False

    use_hvd: bool = True

    identity_extractor_config = {
        "f_3d_checkpoint_path": "/data/useful_ckpt/Deep3DFaceRecon/epoch_20.pth",
        "f_id_checkpoint_path": "/data/useful_ckpt/arcface/ms1mv3_arcface_r100_fp16_backbone.pth",
        "bfm_folder": "/data/useful_ckpt/BFM",
    }

    visualize_interval: int = 100
    plot_interval: int = 10
    max_iters: int = 1000000
    checkpoint_interval: int = 40000

    exp_name: str = "baseline_small"
    log_basedir: str = "/data/logs/hififace/"
    checkpoint_basedir = "/data/checkpoints/hififace"

    def __post_init__(self):
        time_stamp = int(time.time() * 1000)
        self.log_dir = os.path.join(self.log_basedir, f"{self.exp_name}_{time_stamp}")
        self.checkpoint_dir = os.path.join(self.checkpoint_basedir, f"{self.exp_name}_{time_stamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


if __name__ == "__main__":
    tc = TrainConfig()
    print(tc.log_dir)
