import torch
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    Tensorboard 可视化监控类
    """

    def __init__(self, opt):
        """ """
        self.opt = opt  # cache the option
        self.writer = SummaryWriter(log_dir=opt.log_dir)

    def display_current_results(self, iters, visuals_dict):
        """
        Display current images

        Parameters:
        ----------
            visuals (OrderedDict) - - dictionary of images to display
            iters (int) - - the current iteration
        """
        for label, image in visuals_dict.items():
            if image.shape[0] >= 2:
                image = image[0:2, :, :, :]
            self.writer.add_images(str(label), (image * 255.0).to(torch.uint8), global_step=iters, dataformats="NCHW")

    def plot_current_losses(self, iters, loss_dict):
        """
        Display losses on tensorboard

        Parameters:
            iters (int)           -- current iteration
            losses (OrderedDict)  -- training losses stored in the format of (name, torch.Tensor) pairs
        """
        x = iters
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Loss/{k}", v, x)
