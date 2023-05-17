import os
import time
from dataclasses import dataclass

from configs.mode import FaceSwapMode
from configs.singleton import Singleton


@Singleton
@dataclass
class TrainConfig:
    mode = FaceSwapMode.MANY_TO_MANY
    source_name: str = ""

    dataset_index: str = "/data/dataset/faceswap/large.pkl"
    dataset_root: str = "/data/dataset/faceswap"

    batch_size: int = 8
    num_threads: int = 8
    same_rate: float = 0.5
    lr: float = 5e-5
    grad_clip: float = 1000.0

    use_ddp: bool = True

    eye_hm_loss: bool = False
    load_checkpoint = None  # ("/data/checkpoints/hififace/rebuilt_discriminator_SFF_c256_1683367464544", 400000)

    identity_extractor_config = {
        "f_3d_checkpoint_path": "/data/useful_ckpt/Deep3DFaceRecon/epoch_20.pth",
        "f_id_checkpoint_path": "/data/useful_ckpt/arcface/ms1mv3_arcface_r100_fp16_backbone.pth",
        "bfm_folder": "/data/useful_ckpt/BFM",
        "model_ft_path": "/data/useful_ckpt/AWL_detector/WFLW_4HG.pth",
    }

    visualize_interval: int = 100
    plot_interval: int = 100
    max_iters: int = 1000000
    checkpoint_interval: int = 40000

    exp_name: str = "exp_rsff_msd"
    log_basedir: str = "/data/logs/hififace/"
    checkpoint_basedir = "/data/checkpoints/hififace"

    def __post_init__(self):
        time_stamp = int(time.time() * 1000)
        self.log_dir = os.path.join(self.log_basedir, f"{self.exp_name}_{time_stamp}")
        self.checkpoint_dir = os.path.join(self.checkpoint_basedir, f"{self.exp_name}_{time_stamp}")


if __name__ == "__main__":
    tc = TrainConfig()
    print(tc.log_dir)
