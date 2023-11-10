# pytorch-digit-object-detection

## 環境

- Python 3.9
- torch 1.13.1
- torchvision 0.14.1

## 安裝

1. 安裝 pytorch，可至 [PyTorch 官網](https://pytorch.org/) 根據對應版本產生對應的安裝指令。
2. 安裝 Python 套件

```shell
$ pip install -r requirements.txt
```

## 使用說明

- 辨識數字

```shell
python recognize.py -i demo/demo.jpg \
                    -w weights/yolov5_digit.pt \
                    -d [cpu|cuda]
```

| 參數名稱              | 型態   | 必填 | 預設值                    | 說明                  | 備註 |
| --------------------- | ------ | ---- | ------------------------- | --------------------- | ---- |
| `-i`, `--img_path`    | String | Y    |                           | 影像路徑              |      |
| `-w`, `--weight_path` | String | N    | `weights/yolov5_digit.pt` | 模型權重檔路徑        |      |
| `-d`, `--device`      | String | N    | `cpu`                     | 推論設備，cpu 或 cuda |      |
