import sys

sys.path.insert(0, 'ai_models/yolov5_model')

import numpy as np
import torch
from models.common import DetectMultiBackend
from typing import Tuple
from yolo_utils.general import check_img_size, non_max_suppression, scale_boxes, Profile
from yolo_utils.torch_utils import select_device
from yolo_utils.augmentations import letterbox


class YOLOv5Wrapper:

    def __init__(self,
                 weight_path: str,
                 device: str = 'cpu',
                 img_size: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 max_detection: int = 1000) -> None:
        """初始化

        Args:
            weight_path (str): 權重路徑
            device (str, optional): 推論裝置. Defaults to 'cpu'.
            img_size (Tuple[int, int], optional): 模型輸入影像尺寸. Defaults to (640, 640).
            conf_thres (float, optional): 信心值. Defaults to 0.25.
            iou_thres (float, optional): IoU 閾值. Defaults to 0.45.
            max_detection (int, optional): 最大偵測框數量. Defaults to 1000.
        """

        # 1. 載入模型
        # NOTE: 如果是使用 CUDA 的話要指定顯卡位置，預設使用第1張卡(0)
        device = 'cuda:0' if device == 'cuda' else device
        device = select_device(device)
        model = DetectMultiBackend(weight_path, device=device)

        # 2. 設定前處理
        stride, names, pt = model.stride, model.names, model.pt
        img_size = check_img_size(img_size, s=stride)

        # 3. 模型預熱
        batch_size = 1
        model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3,
                            *img_size))  # warmup

        self.device = device
        self.model = model
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_detection = max_detection

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """前處理

        Args:
            img (np.ndarray): 影像

        Returns:
            np.ndarray: 前處理後的影像
        """

        img = letterbox(img, self.img_size,
                        stride=self.model.stride)[0]  # padded resize
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous

        return img

    def predict_img(self,
                    img: np.ndarray) -> Tuple[int, int, int, int, int, float]:
        """預測單張影像

        Args:
            img (np.ndarray): 影像

        Returns:
            Tuple[int, int, int, int, int, float]: 預測結果[分類索引, x1, y1, x2, y2, 信心值]
        """

        # 1. 前處理
        img_input = self.preprocess(img)

        # 2. 推論模型
        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            img_input = torch.from_numpy(img_input).to(self.device)
            img_input = img_input.half(
            ) if self.model.fp16 else img_input.float()  # uint8 to fp16/32
            img_input /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img_input.shape) == 3:
                img_input = img_input[None]  # expand for batch dim

        with dt[1]:
            pred = self.model(img_input)

        # 3. NMS
        with dt[2]:
            pred = non_max_suppression(pred,
                                       self.conf_thres,
                                       self.iou_thres,
                                       max_det=self.max_detection)

        # 4. 整理結果資料
        predictions: Tuple[int, int, int, int, int, float] = []
        for det in pred:

            if len(det):

                # 還原原始座標
                det[:, :4] = scale_boxes(img_input.shape[2:], det[:, :4],
                                         img.shape).round()
                for *xyxy, conf, cls in reversed(det):

                    label_idx = int(cls.item())
                    x1, y1, x2, y2 = xyxy
                    x1 = int(x1.item())
                    y1 = int(y1.item())
                    x2 = int(x2.item())
                    y2 = int(y2.item())
                    conf = conf.item()

                    predictions.append([label_idx, x1, y1, x2, y2, conf])

        return predictions
