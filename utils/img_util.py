import cv2
import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple


def np_to_pil(img: np.ndarray) -> Image:
    """Numpy ndarray 轉成 Pillow Image

    Args:
        img (np.ndarray): numpy array 影像

    Returns:
        Image: Pillow 影像
    """

    # 1. 調整通道(BGR -> RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)


def pil_to_np(img: Image) -> np.ndarray:
    """Numpy ndarray 轉成 Pillow Image

    Args:
        img (Image): Pillow 影像

    Returns:
        np.ndarray: numpy array 影像
    """

    # 1. 轉為 ndarray
    img_np = np.asarray(img)

    # 2. 調整通道(RGB -> BGR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_np


def load_img(img_path: str) -> np.ndarray:
    """載入圖片

    Args:
        img_path (str): 影像路徑

    Raises:
        FileNotFoundError: 檔案不存在錯誤

    Returns:
        np.ndarray: 影像
    """

    # 1. 檢查上層目錄是否存在
    if not Path(img_path).exists():
        raise FileNotFoundError

    # 2. 讀取圖片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

    return img


def save_img(img: np.ndarray, save_path: str) -> None:
    """儲存圖片

    Args:
        img (np.ndarray): 影像物件
        save_path (str): 存檔路徑
    """

    # 1. 檢查檔案是否存在，如果不存在就建立一個
    if not Path(save_path).parent.exists():
        os.makedirs(str(Path(save_path).parent))

    # 2. 存檔
    cv2.imwrite(save_path, img)

    return None


def draw_box(img: np.ndarray,
             boxes: List[List[int]],
             color: Tuple[int] = (255, 255, 255),
             bgr2rgb: bool = True) -> np.ndarray:
    """繪製方框

    Args:
        img (np.ndarray): 影像
        boxes (List[List[int]]): 方框列表
        color (Tuple[int], optional): 顏色(B, G, R). Defaults to (255, 255, 255).
        bgr2rgb (bool, optional): 交換色彩通道(BGR->RGB). Defaults to True.

    Returns:
        np.ndarray: 繪製後的影像
    """

    # 1. 複製一份圖片
    img_copy = img.copy()

    # 2. 交換色彩通道
    if bgr2rgb:
        b, g, r = color
        color = (r, g, b)

    # 3. 繪圖
    for box in boxes:
        img_copy = cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]),
                                 color,
                                 thickness=2)

    return img_copy


def draw_bbox(img: np.ndarray,
              bboxes: Tuple[int, int, int, int, int, float],
              box_color: Tuple[int, int, int] = (255, 0, 0),
              font_color: Tuple[int, int, int] = (255, 255, 255),
              bgr2rgb: bool = True) -> np.ndarray:
    """繪製標記框

    Args:
        img (np.ndarray): 影像
        bboxes (Tuple[int, int, int, int, int, float]): 標記框列表(x_start, y_start, x_end, y_end, label, conf)
        box_color (Tuple[int, int, int], optional): 標記框顏色. Defaults to (255, 0, 0).
        font_color (Tuple[int, int, int], optional): 字體顏色. Defaults to (255, 255, 255).
        bgr2rgb (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: 繪製後的影像
    """

    # 1. 複製一份圖片
    img_copy = img.copy()

    # 2. 交換色彩通道
    if bgr2rgb:
        b, g, r = box_color
        box_color = (r, g, b)

        b, g, r = font_color
        font_color = (r, g, b)

    # 3. 繪圖
    box_thickness = 1
    for bbox in bboxes:

        # 取得資料
        x_start, y_start, x_end, y_end, label, conf = bbox

        # 標記框
        img_copy = cv2.rectangle(img_copy, (x_start, y_start), (x_end, y_end),
                                 box_color, box_thickness)

        # 設定標籤資料
        label_str = f'{label}'
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        label_size = cv2.getTextSize(label_str, font_face, font_scale,
                                     font_thickness)

        # 繪製標籤
        lbx_start = x_start
        lby_start = max(y_start - label_size[0][1], 0)
        lbx_end = max(x_start + label_size[0][0], 0)
        lby_end = y_start
        img_copy = cv2.rectangle(img_copy, (lbx_start, lby_start),
                                 (lbx_end, lby_end), box_color, cv2.FILLED)
        img_copy = cv2.putText(img_copy, label_str, (lbx_start, lby_end),
                               font_face, font_scale, font_color,
                               font_thickness, cv2.LINE_AA)

    return img_copy


def smart_resize(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """智慧縮放圖片(自動決定插值方式)

    Args:
        img (np.ndarray): 影像
        img_size (Tuple[int, int]): 影像尺寸(寬、高)

    Returns:
        np.ndarray: 縮放後的影像
    """

    # 1. 計算插值方式
    img_h, img_w, _ = img.shape
    target_w, target_h = target_shape

    larger_h = target_h >= img_h
    larger_w = target_w >= img_w
    interpolation = cv2.INTER_CUBIC if larger_h and larger_w else cv2.INTER_AREA

    # 2. 修改尺寸
    img_resize = cv2.resize(img, target_shape, interpolation=interpolation)

    return img_resize


def center_fill_border(img: np.ndarray, target_size: Tuple[int,
                                                           int]) -> np.ndarray:
    """填充邊界

    Args:
        img (np.ndarray): 影像尺寸
        target_size (Tuple[int, int]): 目標尺寸(長、寬)

    Returns:
        np.ndarray: 已填充的圖片
    """

    # 1. 計算邊界填充量
    target_h, target_w = target_size
    img_h, img_w = img.shape[:2]

    top = round((target_h - img_h) / 2)
    bottom = max(0, target_h - top - img_h)
    left = round((target_w - img_w) / 2)
    right = max(0, target_w - left - img_w)

    # 2. 填充邊界
    img_filled = cv2.copyMakeBorder(img,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))

    return img_filled
