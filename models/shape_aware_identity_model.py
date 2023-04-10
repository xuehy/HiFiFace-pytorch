import torch
import torch.nn as nn
import torch.nn.functional as F

from arcface_torch.backbones.iresnet import iresnet100
from Deep3DFaceRecon_pytorch.models.networks import ReconNetWrapper


class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self, identity_extractor_config):
        """
        Shape Aware Identity Extractor
        Parameters:
        ----------
        identity_extractor_config: Dict[str, str]
        必须包含以下内容：
            f_3d_checkpoint_path: str
                3D人脸重建模型路径，如"model/Deep3DFaceRecon_pytorch/checkpoints/epoch_20.pth"
            f_id_checkpoint_path: str
                arcface人脸识别模型路径
                非官方实现用的是https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215585&cid=4A83B6B633B029CC/backbone.pth
        """
        super(ShapeAwareIdentityExtractor, self).__init__()
        f_3d_checkpoint_path = identity_extractor_config["f_3d_checkpoint_path"]
        f_id_checkpoint_path = identity_extractor_config["f_id_checkpoint_path"]
        # 3D人脸重建模型
        self.f_3d = ReconNetWrapper(net_recon="resnet50", use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(f_3d_checkpoint_path, map_location="cpu")["net_recon"])
        self.f_3d.eval()

        # 人脸识别模型
        self.f_id = iresnet100(pretrained=False, fp16=False)
        self.f_id.load_state_dict(torch.load(f_id_checkpoint_path, map_location="cpu"))
        self.f_id.eval()

    def forward(self, i_source, i_target):
        """
        Parameters:
        -----------
        i_source: torch.Tensor, shape (B, 3, H, W), in range [0, 1], source face image
        i_target: torch.Tensor, shape (B, 3, H, W), in range [0, 1], target face image

        Returns:
        --------
        v_sid: torch.Tensor, fused shape and id features
        """
        # regress 3DMM coefficients
        c_s = self.f_3d(i_source)
        c_t = self.f_3d(i_target)

        # generate a new 3D face model: source's identity + target's posture and expression
        # from https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/f221678d4b49ca35f1275ba60f721ecb38a2cd19/models/networks.py#L85
        c_fuse = torch.cat((c_s[:, :80], c_t[:, 80:]), dim=1)

        # extract source face identity feature
        v_id = F.normalize(self.f_id(F.interpolate((i_source - 0.5) / 0.5, size=112, mode="bicubic")), dim=-1, p=2)

        # concat new shape feature and source identity
        v_sid = torch.cat((c_fuse, v_id), dim=1)
        return v_sid
