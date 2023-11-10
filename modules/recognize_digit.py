import numpy as np
from collections import defaultdict
from dto.digit_prediction import DigitPrediction
from modules.yolov5_wrapper import YOLOv5Wrapper
from typing import Dict, List, Optional, Tuple, Set


class DigitRecognizer:

    def __init__(self,
                 weight_path: str,
                 device: Optional[str] = 'cpu',
                 threshold: Optional[float] = 0.5,
                 area_percentage: Optional[float] = 0.75,
                 iou_threshold: Optional[float] = 0.78) -> None:

        # 1. 設定參數
        self.weight_path = weight_path
        self.device = device
        self.threshold = threshold
        self.area_percentage = area_percentage
        self.iou_threshold = iou_threshold

        # 2. 建立模型 Wrapper
        self.wrapper = YOLOv5Wrapper(weight_path, device)

    def filter_by_y(self, dps: List[DigitPrediction]) -> List[DigitPrediction]:

        # 1. 計算標記框在 y 值的共同範圍
        idx_to_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx_a, dp_a in enumerate(dps):
            for idx_b, dp_b in enumerate(dps):
                if idx_a == idx_b:
                    continue

                ay_min = round(dp_a.y_min)
                ay_max = round(dp_a.y_max)
                by_min = round(dp_b.y_min)
                by_max = round(dp_b.y_max)

                a_ys = [_ for _ in range(ay_min, ay_max + 1)]
                b_ys = [_ for _ in range(by_min, by_max + 1)]

                union_ys = set(a_ys) & set(b_ys)
                if not union_ys:
                    continue

                idx_to_idxs[idx_a].append(idx_b)

        # 2. 計算相同 y 值的標記框群
        idxs_list: List[Set[int]] = []
        for k, v in idx_to_idxs.items():

            idxs = []
            idxs.append(k)
            idxs.extend(v)

            idxs = set(idxs)

            # 檢查目前的標記框群組是否已經存在
            if not idxs_list:
                idxs_list.append(idxs)
                continue

            is_exist = False
            for _ in idxs_list:
                if _ == idxs:
                    is_exist = True
                    continue

            if not is_exist:
                idxs_list.append(idxs)

        # 3. 按照群組內的標記框數量排序(由多到少)
        idxs_list = [list(_) for _ in idxs_list]
        idxs_list.sort(key=lambda x: len(x), reverse=True)

        # 檢查是否有共同標記框
        if not idxs_list:
            return dps

        # 4. 選擇數量最多的共同標記框
        most_idxs = idxs_list[0]
        most_idxs.sort()

        new_dps: List[DigitPrediction] = []
        for _ in most_idxs:
            new_dps.append(dps[_])

        return new_dps

    def is_suitable_area(self,
                         dp1: DigitPrediction,
                         dp2: DigitPrediction,
                         area_percentage: float = 0.8) -> bool:

        # 1. 取得座標資料
        x1 = dp1.x_min
        y1 = dp1.y_min
        x2 = dp1.x_max
        y2 = dp1.y_max

        x3 = dp2.x_min
        y3 = dp2.y_min
        x4 = dp2.x_max
        y4 = dp2.y_max

        # 2. 計算標記框面積
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)

        if area1 == 0 or area2 == 0:
            return False

        # 3. 判斷兩個標記框面積的比值
        large_area = max(area1, area2)
        small_area = min(area1, area2)
        if small_area / large_area < area_percentage:
            return False

        return True

    def calculate_iou(self, dp1: DigitPrediction,
                      dp2: DigitPrediction) -> float:

        # 1. 取得座標資料
        x1 = dp1.x_min
        y1 = dp1.y_min
        x2 = dp1.x_max
        y2 = dp1.y_max

        x3 = dp2.x_min
        y3 = dp2.y_min
        x4 = dp2.x_max
        y4 = dp2.y_max

        a_inter = max(x1, x3)
        b_inter = max(y1, y3)
        c_inter = min(x2, x4)
        d_inter = min(y2, y4)

        # 2. 檢查交集面積是否為0
        if (c_inter < a_inter) or (d_inter < b_inter):
            return 0.0

        # 3. 計算交集與聯集面積
        intersection = (c_inter - a_inter) * (d_inter - b_inter)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = (x4 - x3) * (y4 - y3)
        union = area_a + area_b - intersection

        # 4. 計算IoU
        return intersection / union

    def filter_by_iou(self,
                      dps: List[DigitPrediction],
                      area_percentage: float = 0.75,
                      iou_threshold: float = 0.78) -> List[DigitPrediction]:

        # 1. 透過面積比例與 IoU 過濾標記框
        reserve_idxs = []
        remove_idxs = []
        for idx_a, dp_a in enumerate(dps):
            for idx_b, dp_b in enumerate(dps):

                # 檢查標記框是否相同、是否已經被檢查過
                if idx_a == idx_b:
                    continue

                if idx_a in reserve_idxs or idx_a in remove_idxs:
                    continue

                if idx_b in reserve_idxs or idx_b in remove_idxs:
                    continue

                # 檢查兩個標記框面積是否合適，不會差距太大
                if not self.is_suitable_area(dp_a, dp_b, area_percentage):
                    continue

                # 判斷 IOU 的大小
                iou = self.calculate_iou(dp_a, dp_b)
                if iou < iou_threshold:
                    continue

                # 比較兩標記框的信心值，挑選兩標記框中信心值較高的標記框作為最後的答案
                if dp_a.confidence >= dp_b.confidence:
                    reserve_idxs.append(idx_a)
                    remove_idxs.append(idx_b)
                else:
                    reserve_idxs.append(idx_b)
                    remove_idxs.append(idx_a)

        # 2. 整理最後要使用的標記框
        # NOTE: 最後要使用的標記框 = 沒有重疊、有重疊但是重疊面積未達門檻的標記框
        use_idxs: List[int] = []
        for idx, _ in enumerate(dps):
            if idx in reserve_idxs:
                continue

            if idx in remove_idxs:
                continue

            use_idxs.append(idx)

        # 按照索引大小排序，避免打亂原本的順序
        use_idxs.extend(reserve_idxs)
        use_idxs.sort()

        # 3. 整理資料
        boxes = [dps[_] for _ in use_idxs]

        return boxes

    def to_number(self, digits: List[str]) -> Optional[int]:

        number_str = ''.join(digits)
        return int(number_str) if number_str else None

    def recognize(
            self,
            img: np.ndarray) -> Tuple[Optional[int], List[DigitPrediction]]:

        # 1. 推論模型
        predictions = self.wrapper.predict_img(img)

        # 2. 按照 x 軸由左至右排序預測結果
        predictions.sort(key=lambda x: x[1])

        # 3. 整理資料
        digit_predictions: List[DigitPrediction] = []
        for p in predictions:

            digit_prediction = DigitPrediction(label_idx=p[0],
                                               x_min=p[1],
                                               y_min=p[2],
                                               x_max=p[3],
                                               y_max=p[4],
                                               confidence=p[5])
            digit_predictions.append(digit_prediction)

        # 4. 過濾非在同一水平線上的標記框與重疊的標記框
        filtered_y_dps = self.filter_by_y(digit_predictions)
        filtered_iou_dps = self.filter_by_iou(filtered_y_dps,
                                              self.area_percentage,
                                              self.iou_threshold)

        # 5. 轉為數字
        # NOTE: 2023-07-04 加入其他類別，標籤other，索引為10，所以只取標籤0-9當作0-9的數字使用
        labels = [
            str(_.label_idx) for _ in filtered_iou_dps
            if _.confidence > self.threshold and _.label_idx < 10
        ]
        number = self.to_number(labels)

        return number, digit_predictions
