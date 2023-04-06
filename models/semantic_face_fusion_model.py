import torch.nn as nn
import torch.nn.functional as F

from models.model_blocks import AdaInResBlock
from models.model_blocks import ResBlock
from models.model_blocks import UpSamplingBlock


class SemanticFaceFusionModule(nn.Module):
    def __init__(self):
        """
        Semantic Face Fusion Module
        to preserve lighting and background
        """
        super(SemanticFaceFusionModule, self).__init__()

        self.sigma = ResBlock(256, 256)
        self.low_mask_predict = ResBlock(256, 1)
        self.z_fuse_block = AdaInResBlock(256, 256 + 3)
        self.f_up = UpSamplingBlock()

    def forward(self, target_image, z_enc, z_dec, v_sid):
        """
        Parameters:
        ----------
        target_image: 目标脸图片
        z_enc: 1/4原图大小的low-level encoder feature map
        z_dec: 1/4原图大小的low-level decoder feature map
        v_sid: the 3D shape aware identity vector

        Returns:
        --------
        i_r: re-target image
        i_low: 1/4 size retarget image
        m_r: face mask
        m_low: 1/4 size face mask
        """
        z_enc = self.sigma(z_enc)

        # 估算z_dec对应的人脸 low-level feature mask
        m_low = self.low_mask_predict(z_dec)

        m_low = m_low.float()
        m_low = F.sigmoid(m_low)
        # 计算融合的low-level feature map
        # mask区域使用decoder的low-level特征 + 非mask区域使用encoder的low-level特征
        z_fuse = m_low * z_dec + (1 - m_low) * z_enc

        z_fuse = self.z_fuse_block(z_fuse, v_sid)

        i_low = z_fuse[:, 0:3, ...]
        z_fuse = z_fuse[:, 3:, ...]

        i_low = m_low * i_low + (1 - m_low) * F.interpolate(target_image, scale_factor=0.25)

        i_r, m_r = self.f_up(z_fuse)
        i_r = m_r * i_r + (1 - m_r) * target_image

        return i_r, i_low, m_r, m_low
