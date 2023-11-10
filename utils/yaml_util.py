import yaml


def load_yaml(path: str) -> dict:
    """讀取 yaml 檔

    Args:
        path (str): 檔案路徑

    Returns:
        dict: yaml 字典資料
    """

    # 1. 讀取 yaml 檔
    data = dict()
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data