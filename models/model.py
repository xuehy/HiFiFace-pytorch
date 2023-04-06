import os
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import kornia
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

from arcface_torch.backbones.iresnet import iresnet100
from configs.train_config import TrainConfig
from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel
from Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper
from models.discriminator import MultiscaleDiscriminator
from models.gan_loss import MultiScaleGANLoss
from models.generator import Generator


class HifiFace(nn.Module):
    def __init__(self, identity_extractor_config, is_training=True):
        super(HifiFace, self).__init__()
        self.generator = Generator(identity_extractor_config)
        self.amp = TrainConfig().amp
        # 判别器的定义还不对，可能需要对照论文里面的图片进行修改
        self.discriminator = MultiscaleDiscriminator(3)

        self.is_training = is_training

        if self.is_training:
            self.l1_loss = nn.L1Loss()
            self.loss_fn_vgg = lpips.LPIPS(net="vgg")
            self.adv_loss = MultiScaleGANLoss()

            # 3D人脸重建模型
            self.f_3d = ReconNetWrapper(net_recon="resnet50", use_last_fc=False)
            self.f_3d.load_state_dict(
                torch.load(identity_extractor_config["f_3d_checkpoint_path"], map_location="cpu")["net_recon"]
            )
            self.f_3d.eval()
            self.face_model = ParametricFaceModel(bfm_folder=identity_extractor_config["bfm_folder"])
            self.face_model.to("cpu")

            # 人脸识别模型
            self.f_id = iresnet100(pretrained=False, fp16=False)
            self.f_id.load_state_dict(torch.load(identity_extractor_config["f_id_checkpoint_path"], map_location="cpu"))
            self.f_id.eval()

            self.lambda_adv = 1
            self.lambda_seg = 100
            self.lambda_rec = 20
            self.lambda_cyc = 1
            self.lambda_lpips = 5

            self.lambda_shape = 0.5
            self.lambda_id = 5

            self.dilation_kernel = torch.ones(5, 5)
            # optimizers
            self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=1e-4, betas=[0, 0.999])
            self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=[0, 0.999])
            if self.amp:
                self.scaler_G = GradScaler()
                self.scaler_D = GradScaler()

    def save(self, path, idx=None):
        if idx is None:
            g_path = os.path.join(path, "generator.pth")
            d_path = os.path.join(path, "discriminator.pth")
        else:
            g_path = os.path.join(path, f"generator_{idx}.pth")
            d_path = os.path.join(path, f"discriminator_{idx}.pth")
        torch.save(self.generator.state_dict(), g_path)
        torch.save(self.discriminator.state_dict(), d_path)

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        if self.is_training:
            self.f_3d.to(device)
            self.f_id.to(device)
            self.loss_fn_vgg.to(device)
            self.face_model.to(device)
            self.dilation_kernel = self.dilation_kernel.to(device)
            self.g_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=1e-4, betas=[0, 0.999])
            self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=1e-4, betas=[0, 0.999])

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def inference(self, source_img, target_img):
        with autocast():
            i_r, _, _, _ = self.generator(source_img, target_img)
        return i_r

    def train_forward_generator(self, source_img, target_img, target_mask, same_id_mask):
        """
        训练时候 Generator的loss计算
        Parameters:
        -----------
        source_img: torch.Tensor
        target_img: torch.Tensor
        target_mask: torch.Tensor, [B, 1, H, W]
        same_id_mask: torch.Tensor, [B, 1]

        Returns:
        source_img: torch.Tensor
        target_img: torch.Tensor
        i_r: torch.Tensor
        m_r: torch.Tensor
        loss: Dict[torch.Tensor], contain pairs of loss name and loss values
        """
        same = same_id_mask.unsqueeze(-1).unsqueeze(-1)
        i_r, i_low, m_r, m_low = self.generator(source_img, target_img)
        i_cylce, _, _, _ = self.generator(target_img, i_r)
        d_r = self.discriminator(i_r)

        # SID Loss: shape loss + id loss

        with torch.no_grad():
            c_s = self.f_3d(F.interpolate(source_img, size=224, mode="bilinear"))
            c_t = self.f_3d(F.interpolate(target_img, size=224, mode="bilinear"))
        c_r = self.f_3d(F.interpolate(i_r, size=224, mode="bilinear"))
        c_low = self.f_3d(F.interpolate(i_low, size=224, mode="bilinear"))
        with torch.no_grad():
            c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)
            _, _, _, q_fuse = self.face_model.compute_for_render(c_fuse)
        _, _, _, q_r = self.face_model.compute_for_render(c_r)
        _, _, _, q_low = self.face_model.compute_for_render(c_low)
        with torch.no_grad():
            v_id_i_s = F.normalize(
                self.f_id(F.interpolate((source_img - 0.5) / 0.5, size=112, mode="bilinear")), dim=-1, p=2
            )

        v_id_i_r = F.normalize(self.f_id(F.interpolate((i_r - 0.5) / 0.5, size=112, mode="bilinear")), dim=-1, p=2)
        v_id_i_low = F.normalize(self.f_id(F.interpolate((i_low - 0.5) / 0.5, size=112, mode="bilinear")), dim=-1, p=2)
        loss_shape = self.l1_loss(q_fuse, q_r) + self.l1_loss(q_fuse, q_low)
        loss_shape = torch.clamp(loss_shape, min=0.0, max=10.0)

        inner_product_r = torch.bmm(v_id_i_s.unsqueeze(1), v_id_i_r.unsqueeze(2)).squeeze()
        inner_product_low = torch.bmm(v_id_i_s.unsqueeze(1), v_id_i_low.unsqueeze(2)).squeeze()
        loss_id = self.l1_loss(torch.ones_like(inner_product_r), inner_product_r) + self.l1_loss(
            torch.ones_like(inner_product_low), inner_product_low
        )
        loss_sid = self.lambda_shape * loss_shape + self.lambda_id * loss_id

        # Realism Loss: segmentation loss + reconstruction loss + cycle loss + perceptual loss + adversarial loss

        loss_cycle = self.l1_loss(target_img, i_cylce)

        # dilate target mask
        target_mask = kornia.morphology.dilation(target_mask, self.dilation_kernel)

        loss_segmentation = self.l1_loss(
            F.interpolate(target_mask, scale_factor=0.25, mode="bilinear"), m_low
        ) + self.l1_loss(target_mask, m_r)

        loss_reconstruction = self.l1_loss(i_r * same, target_img * same) + self.l1_loss(
            i_low * same, F.interpolate(target_img, scale_factor=0.25, mode="bilinear") * same
        )

        loss_perceptual = self.loss_fn_vgg(target_img * same, i_r * same).mean()

        loss_adversarial = self.adv_loss(d_r, True, for_discriminator=False)

        loss_realism = (
            self.lambda_adv * loss_adversarial
            + self.lambda_seg * loss_segmentation
            + self.lambda_rec * loss_reconstruction
            + self.lambda_cyc * loss_cycle
            + self.lambda_lpips * loss_perceptual
        )

        loss_generator = loss_sid + loss_realism
        return (
            source_img,
            target_img,
            i_r.detach(),
            m_r.detach(),
            {
                "loss_shape": loss_shape,
                "loss_id": loss_id,
                "loss_sid": loss_sid,
                "loss_cycle": loss_cycle,
                "loss_segmentation": loss_segmentation,
                "loss_reconstruction": loss_reconstruction,
                "loss_perceptual": loss_perceptual,
                "loss_adversarial": loss_adversarial,
                "loss_realism": loss_realism,
                "loss_generator": loss_generator,
            },
        )

    def train_forward_discriminator(self, target_img, i_r):
        """
        训练时候 Discriminator的loss计算
        Parameters:
        -----------
        target_img: torch.Tensor, 目标脸图片
        i_r: torch.Tensor, 换脸结果

        Returns:
        --------
        Dict[str]: contains pair of loss name and loss values
        """
        d_gt = self.discriminator(target_img)
        d_fake = self.discriminator(i_r.detach())
        loss_real = self.adv_loss(d_gt, True)
        loss_fake = self.adv_loss(d_fake, False)
        loss_discriminator = loss_real + loss_fake
        return {"loss_real": loss_real, "loss_fake": loss_fake, "loss_discriminator": loss_discriminator}

    def forward(
        self,
        source_img: torch.Tensor,
        target_img: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        same_id_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[Dict, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        模型的forward
        训练模式下执行一次训练，并返回loss信息和结果
        推理模式下，直接执行换脸操作，并返回结果
        Parameters:
        -----------
        source_img: torch.Tensor, source face 图像
        target_img: torch.Tensor, target face 图像
        target_mask: Optional[torch.Tensor], target face mask, 训练时需要
        same_id_mask: Optional[torch.Tensor], same id mask, 标识source 和 target是否是同个人，训练时需要

        Returns:
        --------
        Union[Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        loss_dict, source_img, target_img, m_r(预测的mask), i_r（换脸结果） | i_r
        """
        if self.training:
            if self.amp:
                with autocast():
                    src_img, tgt_img, i_r, m_r, loss_G_dict = self.train_forward_generator(
                        source_img, target_img, target_mask, same_id_mask
                    )
            else:
                src_img, tgt_img, i_r, m_r, loss_G_dict = self.train_forward_generator(
                    source_img, target_img, target_mask, same_id_mask
                )
            loss_G = loss_G_dict["loss_generator"]
            self.g_optimizer.zero_grad()
            if self.amp:
                self.scaler_G.scale(loss_G).backward()
                self.scaler_G.step(self.g_optimizer)
                self.scaler_G.update()
            else:
                loss_G.backward()
                self.g_optimizer.step()

            loss_D_dict = self.train_forward_discriminator(tgt_img, i_r)
            loss_D = loss_D_dict["loss_discriminator"]
            self.d_optimizer.zero_grad()
            if self.amp:
                self.scaler_D.scale(loss_D).backward()
                self.scaler_D.step(self.d_optimizer)
                self.scaler_D.update()
            else:
                loss_D.backward()
                self.d_optimizer.step()

            total_loss_dict = {}
            total_loss_dict.update(loss_G_dict)
            total_loss_dict.update(loss_D_dict)
            return total_loss_dict, {
                "source face": src_img,
                "target_face": tgt_img,
                "swapped face": i_r,
                "pred face mask": m_r,
            }
        else:
            return self.inference(source_img, target_img)


if __name__ == "__main__":
    import torch
    import cv2
    from configs.train_config import TrainConfig

    identity_extractor_config = TrainConfig().identity_extractor_config

    model = HifiFace(identity_extractor_config, is_training=True)

    src = cv2.imread("/home/xuehongyang/data/test1.jpg")
    tgt = cv2.imread("/home/xuehongyang/data/test2.jpg")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (256, 256))
    tgt = cv2.resize(tgt, (256, 256))
    src = src.transpose(2, 0, 1)[None, ...]
    tgt = tgt.transpose(2, 0, 1)[None, ...]
    source_img = torch.from_numpy(src).float() / 255.0
    target_img = torch.from_numpy(tgt).float() / 255.0
    same_id_mask = torch.Tensor([1]).unsqueeze(0)
    tgt_mask = target_img[:, 0, :, :].unsqueeze(1)
    if torch.cuda.is_available():
        model.to("cuda:3")
        source_img = source_img.to("cuda:3")
        target_img = target_img.to("cuda:3")
        tgt_mask = tgt_mask.to("cuda:3")
        same_id_mask = same_id_mask.to("cuda:3")
        source_img = source_img.repeat(16, 1, 1, 1)
        target_img = target_img.repeat(16, 1, 1, 1)
        tgt_mask = tgt_mask.repeat(16, 1, 1, 1)
        same_id_mask = same_id_mask.repeat(16, 1)
    while True:
        x = model(source_img, target_img, tgt_mask, same_id_mask)
        print(x[0]["loss_generator"])
