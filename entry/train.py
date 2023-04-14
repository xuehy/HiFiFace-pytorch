import os
import sys

import torch
from loguru import logger

from configs.train_config import TrainConfig
from data.dataset import TrainDatasetDataLoader
from models.model import HifiFace
from utils.visualizer import Visualizer

use_ddp = TrainConfig().use_ddp
if use_ddp:

    import torch.distributed as dist

    def setup():
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group("nccl")  # , rank=rank, world_size=world_size)
        return dist.get_rank()

    def cleanup():
        dist.destroy_process_group()


def train():
    rank = 0
    if use_ddp:
        rank = setup()
    device = torch.device(f"cuda:{rank}")
    logger.info(f"use device {device}")

    opt = TrainConfig()
    dataloader = TrainDatasetDataLoader()
    dataset_length = len(dataloader)
    logger.info(f"Dataset length: {dataset_length}")

    model = HifiFace(
        opt.identity_extractor_config, is_training=True, device=device, load_checkpoint=opt.load_checkpoint
    )
    model.train()

    logger.info("model initialized")
    visualizer = None
    ckpt = False
    if not opt.use_ddp or rank == 0:
        visualizer = Visualizer(opt)
        ckpt = True

    total_iter = 0
    epoch = 0
    while True:
        if opt.use_ddp:
            dataloader.train_sampler.set_epoch(epoch)
        for data in dataloader:
            source_image = data["source_image"].to(device)
            target_image = data["target_image"].to(device)
            targe_mask = data["target_mask"].to(device)
            same = data["same"].to(device)
            loss_dict, visual_dict = model.optimize(source_image, target_image, targe_mask, same)

            total_iter += 1

            if total_iter % opt.visualize_interval == 0 and visualizer is not None:
                visualizer.display_current_results(total_iter, visual_dict)

            if total_iter % opt.plot_interval == 0 and visualizer is not None:
                visualizer.plot_current_losses(total_iter, loss_dict)
                logger.info(f"Iter: {total_iter}")
                for k, v in loss_dict.items():
                    logger.info(f" {k}: {v}")
                logger.info("=" * 20)

            if total_iter % opt.checkpoint_interval == 0 and ckpt:
                logger.info(f"Saving model at iter {total_iter}")
                model.save(opt.checkpoint_dir, total_iter)

            if total_iter > opt.max_iters:
                logger.info(f"Maximum iterations exceeded. Stopping training.")
                if ckpt:
                    model.save(opt.checkpoint_dir, total_iter)
                if use_ddp:
                    cleanup()
                sys.exit(0)
        epoch += 1


if __name__ == "__main__":
    if use_ddp:
        # CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 -m entry.train
        os.environ["OMP_NUM_THREADS"] = "1"
        n_gpus = torch.cuda.device_count()
        train()
    else:
        train()
