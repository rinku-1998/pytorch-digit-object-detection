from modules.recognize_digit import DigitRecognizer
from pprint import pprint
from utils.img_util import load_img


def run(img_path: str, weight_path: str, device: str) -> None:
    """開始辨識數字

    Args:
        img_path (str): 影像路徑
        weight_path (str): 權重路徑
        device (str): 推論設備(cpu, cuda)
    """

    # 1. 讀取圖片
    img = load_img(img_path)

    # 2. 建立數字辨識物件
    dr = DigitRecognizer(weight_path=weight_path, device=device)

    # 3. 辨識數字
    number, digit_predictions = dr.recognize(img)
    print(f'Predicted number: {number}')
    print('Bounded boxes:')
    pprint(digit_predictions)


if __name__ == '__main__':

    # 1. 設定參數
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--img_path',
                        type=str,
                        required=True,
                        help='Path to image')
    parser.add_argument('-w',
                        '--weight_path',
                        type=str,
                        default=r'weights/yolov5_digit.pt',
                        help='Path to weight')
    parser.add_argument('-d',
                        '--device',
                        type=str,
                        default='cpu',
                        help='Device to inference')

    args = parser.parse_args()

    # 2. 辨識
    run(args.img_path, args.weight_path, args.device)
