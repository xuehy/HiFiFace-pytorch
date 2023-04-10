import torch
from loguru import logger

from configs.train_config import TrainConfig
from data.dataset import TrainDatasetDataLoader
from models.model import HifiFace
from utils.visualizer import Visualizer

use_hvd = TrainConfig().use_hvd
if use_hvd:
    import horovod.torch as hvd

    hvd.init()


def train():
    if use_hvd:
        device = torch.device(f"cuda:{hvd.local_rank()}")
        logger.info(f"use device {device}")
    else:
        device = torch.device(f"cuda:0")

    opt = TrainConfig()
    dataloader = TrainDatasetDataLoader()
    dataset_length = len(dataloader)
    logger.info(f"Dataset length: {dataset_length}")

    model = HifiFace(opt.identity_extractor_config, is_training=True)
    model.to(device)
    model.train()

    visualizer = None
    ckpt = False
    if not opt.use_hvd or hvd.rank() == 0:
        visualizer = Visualizer(opt)
        ckpt = True

    total_iter = 0
    while True:
        for data in dataloader:
            source_image = data["source_image"].to(device)
            target_image = data["target_image"].to(device)
            targe_mask = data["target_mask"].to(device)
            same = data["same"].to(device)
            loss_dict, visual_dict = model(source_image, target_image, targe_mask, same)

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
                break


if __name__ == "__main__":
    train()
    # horovodrun -np 4 python train.py
