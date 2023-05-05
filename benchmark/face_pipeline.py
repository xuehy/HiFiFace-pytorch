import time
from pathlib import Path
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import cv2
import numpy as np
import torch
from insightface.model_zoo import model_zoo
from skimage import transform as skt

# ---frontal
src = np.array(
    [
        [39.730, 51.138],
        [72.270, 51.138],
        [56.000, 68.493],
        [42.463, 87.010],
        [69.537, 87.010],
    ],
    dtype=np.float32,
)


class alignFace:
    def __init__(self) -> None:
        self.src_map = src

    def estimate_norm(self, lmk, image_size=112):
        assert lmk.shape == (5, 2)
        tform = skt.SimilarityTransform()
        src_ = self.src_map * image_size / 112
        tform.estimate(lmk, src_)
        M = tform.params[0:2, :]
        return M

    def align_face(
        self, img: np.ndarray, key_points: np.ndarray, crop_size: int
    ) -> Tuple[Iterable[np.ndarray], Iterable[np.ndarray]]:
        transform_matrix = self.estimate_norm(key_points, crop_size)
        align_img = cv2.warpAffine(img, transform_matrix, (crop_size, crop_size), borderValue=0.0)
        return align_img, transform_matrix


class Detection(NamedTuple):
    bbox: Optional[np.ndarray]
    score: Optional[np.ndarray]
    key_points: Optional[np.ndarray]


class FaceDetector:
    def __init__(
        self,
        model_path: Path,
        det_thresh: float = 0.5,
        det_size: Tuple[int, int] = (640, 640),
        mode: str = "None",
        device: str = "cuda",
    ):
        self.det_thresh = det_thresh
        self.mode = mode
        self.device = device
        self.handler = model_zoo.get_model(str(model_path))
        ctx_id = -1 if device == "cpu" else 0
        self.handler.prepare(ctx_id, input_size=det_size)

    def __call__(self, img: np.ndarray, max_num: int = 0) -> Detection:
        bboxes, kpss = self.handler.detect(img,  max_num=max_num, metric="default")
        if bboxes.shape[0] == 0:
            return Detection(None, None, None)
        return Detection(bboxes[..., :-1], bboxes[..., -1], kpss)


def tensor2img(tensor):
    tensor = tensor.detach().cpu().numpy()
    img = tensor.transpose(0, 2, 3, 1)[0]
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return img


def inverse_transform_batch(mat: torch.Tensor) -> torch.Tensor:
    # inverse the Affine transformation matrix
    inv_mat = torch.zeros_like(mat).cuda()
    div1 = mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 0]
    inv_mat[:, 0, 0] = mat[:, 1, 1] / div1
    inv_mat[:, 0, 1] = -mat[:, 0, 1] / div1
    inv_mat[:, 0, 2] = -(mat[:, 0, 2] * mat[:, 1, 1] - mat[:, 0, 1] * mat[:, 1, 2]) / div1
    div2 = mat[:, 0, 1] * mat[:, 1, 0] - mat[:, 0, 0] * mat[:, 1, 1]
    inv_mat[:, 1, 0] = mat[:, 1, 0] / div2
    inv_mat[:, 1, 1] = -mat[:, 0, 0] / div2
    inv_mat[:, 1, 2] = -(mat[:, 0, 2] * mat[:, 1, 0] - mat[:, 0, 0] * mat[:, 1, 2]) / div2
    return inv_mat
