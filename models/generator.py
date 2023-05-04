import torch
import torch.nn as nn

from models.init_weight import init_net
from models.model_blocks import AdaInResBlock
from models.model_blocks import ResBlock
from models.semantic_face_fusion_model import SemanticFaceFusionModule
from models.shape_aware_identity_model import ShapeAwareIdentityExtractor


class Encoder(nn.Module):
    """
    Hififace encoder part
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.channel_list = [64, 128, 256, 512, 512, 512, 512, 512]
        self.down_sample = [True, True, True, True, True, False, False]

        self.block_list = nn.ModuleList()

        for i in range(7):
            self.block_list.append(
                ResBlock(self.channel_list[i], self.channel_list[i + 1], down_sample=self.down_sample[i])
            )

    def forward(self, x):
        x = self.conv_first(x)
        z_enc = None

        for i in range(7):
            x = self.block_list[i](x)
            if i == 1:
                z_enc = x
        return z_enc, x


class Decoder(nn.Module):
    """
    Hififace decoder part
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.block_list = nn.ModuleList()
        self.channel_list = [512, 512, 512, 512, 512, 256]
        self.up_sample = [False, False, True, True, True]

        for i in range(5):
            self.block_list.append(
                AdaInResBlock(self.channel_list[i], self.channel_list[i + 1], up_sample=self.up_sample[i])
            )

    def forward(self, x, id_vector):
        """
        Parameters:
        -----------
        x: encoder encoded feature map
        id_vector: 3d shape aware identity vector

        Returns:
        --------
        z_dec
        """
        for i in range(5):
            x = self.block_list[i](x, id_vector)
        return x


class Generator(nn.Module):
    """
    Hififace Generator
    """

    def __init__(self, identity_extractor_config):
        super(Generator, self).__init__()
        self.id_extractor = ShapeAwareIdentityExtractor(identity_extractor_config)
        self.id_extractor.requires_grad_(False)
        self.encoder = init_net(Encoder())
        self.decoder = init_net(Decoder())
        self.sff_module = init_net(SemanticFaceFusionModule())

    def forward(self, i_source, i_target, need_id_grad=False):
        """
        Parameters:
        -----------
        i_source: torch.Tensor, shape (B, 3, H, W), in range [0, 1], source face image
        i_target: torch.Tensor, shape (B, 3, H, W), in range [0, 1], target face image
        need_id_grad: bool, whether to calculate id extractor module's gradient

        Returns:
        --------
        i_r:    torch.Tensor
        i_low:  torch.Tensor
        m_r:    torch.Tensor
        m_low:  torch.Tensor
        """
        if need_id_grad:
            shape_aware_id_vector = self.id_extractor(i_source, i_target)
        else:
            with torch.no_grad():
                shape_aware_id_vector = self.id_extractor(i_source, i_target)
        z_enc, x = self.encoder(i_target)
        z_dec = self.decoder(x, shape_aware_id_vector)

        i_r, i_low, m_r, m_low = self.sff_module(i_target, z_enc, z_dec, shape_aware_id_vector)

        return i_r, i_low, m_r, m_low
