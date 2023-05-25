# -*- coding: utf-8 -*-
"""
@File : scrfd
@Description: scrfd人脸检测
@Author: Yang Jian
@Contact: lian01110@outlook.com
@Time: 2022/2/25 10:31
@IDE: PYTHON
@REFERENCE: https://github.com/yangjian1218
"""
from __future__ import division

import datetime
import os
import os.path as osp
import sys

import cv2
import numpy as np
import onnx
import onnxruntime
from cv2 import KeyPoint

# import face_align


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file=None, session=None, device="cuda", det_thresh=0.5):
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            if device == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                providers = ["CUDAExecutionProvider"]
            self.session = onnxruntime.InferenceSession(self.model_file, providers=providers)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = det_thresh
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        # print("input_shape:",input_shape)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        # print('image_size:', self.image_size)
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        # print("input_name:",self.input_name)
        # print("output_name:",self.output_names)
        self.input_mean = 127.5
        self.input_std = 127.5
        # assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def init_det_threshold(self, det_threshold):
        """
        单独设置人脸检测阈值
        :param det_threshold: 人脸检测阈值
        :return:
        """
        self.det_thresh = det_threshold

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_threshold = kwargs.get("nms_threshold", None)
        if nms_threshold is not None:
            self.nms_threshold = nms_threshold
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in scrfd model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, threshold=0.6, swap_rb=True):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        # print('input_size:',input_size)
        blob = cv2.dnn.blobFromImages(
            [img], 1.0 / self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=swap_rb
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # print("net_outs:::",net_outs[0])
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc  # 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            # print("scores:",scores)
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            # print(anchor_centers.shape,bbox_preds.shape,scores.shape,kps_preds.shape)
            pos_inds = np.where(scores >= threshold)[0]
            # print("pos_inds:",pos_inds)
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        # print("....:",bboxes_list)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, det_thresh=None, metric="default", swap_rb=True):
        """

        :param img: 原始图像
        :param input_size:  输入尺寸,元组或者列表
        :param max_num: 返回人脸数量, 如果为0,表示所有,
        :param det_thresh: 人脸检测阈值,
        :param metric: 排序方式,默认为面积+中心偏移, "max"为面积最大排序
        :param swap_rb: 是否进行r b通道转换, 如果传入的是bgr格式图片,则需要为True
        :return:
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        # resize方法选择,缩小选择cv2.INTER_AREA , 放大选择cv2.INTER_LINEAR
        resize_interpolation = cv2.INTER_AREA if img.shape[0] >= input_size[0] else cv2.INTER_LINEAR
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=resize_interpolation)
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        if det_thresh == None:
            det_thresh = self.det_thresh
        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh, swap_rb)
        # print("====",len(scores_list),len(bboxes_list),len(kpss_list))
        # print("scores_list:",scores_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


if __name__ == "__main__":

    detector = SCRFD(
        model_file="/mnt/c/yangguo/useful_ckpt/face_detector/face_detector_scrfd_10g_bnkps.onnx", device="cpu"
    )
    # detector.prepare()
    img_path = "/mnt/c/yangguo/hififace_infer/src_image/boy.jpg"
    img = cv2.imread(img_path)
    ta = datetime.datetime.now()
    cycle = 100
    # for i in range(cycle):
    bboxes, kpss = detector.detect(img, input_size=(640, 640))  # 得到box跟关键点
    # print("bboxes:",bboxes,"\nkpss:",kpss)
    tb = datetime.datetime.now()
    print("all cost:", (tb - ta).total_seconds() * 1000)
    print(img_path, bboxes.shape)
    if kpss is not None:
        print(kpss.shape)
    # todo 画图
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                kp = kp.astype(np.int32)
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
    # cv2.namedWindow("img", 2)
    cv2.imwrite("./img.jpg", img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
