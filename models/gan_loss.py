import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGANLoss(nn.Module):
    def __init__(
        self, gan_mode="hinge", target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None
    ):
        super(MultiScaleGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == "ls":
            pass
        elif gan_mode == "original":
            pass
        elif gan_mode == "w":
            pass
        elif gan_mode == "hinge":
            pass
        else:
            raise ValueError("Unexpected gan_mode {}".format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, inputs, target_is_real, for_discriminator=True):
        if self.gan_mode == "original":  # cross entropy loss
            target_tensor = self.get_target_tensor(inputs, target_is_real)
            loss = F.binary_cross_entropy_with_logits(inputs, target_tensor)
            return loss
        elif self.gan_mode == "ls":
            target_tensor = self.get_target_tensor(inputs, target_is_real)
            return F.mse_loss(inputs, target_tensor)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(inputs - 1, self.get_zero_tensor(inputs))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-inputs - 1, self.get_zero_tensor(inputs))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(inputs)
            return loss
        else:
            # wgan
            if target_is_real:
                return -inputs.mean()
            else:
                return inputs.mean()

    def __call__(self, inputs, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(inputs, list):
            loss = 0
            for pred_i in inputs:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(inputs)
        else:
            return self.loss(inputs, target_is_real, for_discriminator)
