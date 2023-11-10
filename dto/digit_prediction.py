from dataclasses import dataclass, field


@dataclass
class DigitPrediction:

    label_idx: int = field(init=True)  # 標籤索引
    x_min: float = field(init=True)  # x 最小值
    y_min: float = field(init=True)  # y 最小值
    x_max: float = field(init=True)  # x 最大值
    y_max: float = field(init=True)  # y 最大值
    confidence: float = field(init=True)  # 信心值
