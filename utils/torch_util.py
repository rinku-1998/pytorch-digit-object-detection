import numpy as np
import torch


def auto_device() -> str:
    """自動判斷要使用哪種裝置

    Returns:
        str: 裝置名稱(cpu、cuda)
    """

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Tensor 轉 numpy 陣列

    Args:
        tensor (torch.Tensor): tensor

    Returns:
        np.ndarray: numpy 陣列
    """

    return tensor.cpu().detach().numpy()