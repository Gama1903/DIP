import imageio
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["imread_m, imshow_m,"]


def imread_m(path: str) -> np.ndarray:
    """ͨ��imageio.imread()����ͼƬ��תΪnp.ndarray

        Args:
            path (str): ԴͼƬ����λ��

        Returns:
            np.ndarray: ����np.ndarray��Ԫ������uint8
        """
    return np.array(imageio.imread(path))


def imshow_m(title: str, imgs: tuple, cmaps: list, row: int = 0, col: int = 0):
    """չʾͼƬ

    Args:
        title (str): ͼ�����
        imgs (tuple): ͼƬԪ��
        cmaps (list): plt�Ժ���ͼƬ����չʾͼƬ
        row (int, optional): ָ��row. Defaults to 0.
        col (int, optional): ָ��col. Defaults to 0.
    """
    try:
        if row == 0 and col != 0:
            row = np.ceil(len(imgs) / col).astyle("uint8")
        elif row != 0 and col == 0:
            col = np.ceil(len(imgs) / row).astyle("uint8")
        elif row * col < len(imgs):
            # �����Է�������ʽȥչʾͼƬ
            row = np.ceil(np.sqrt(len(imgs))).astype("uint8")
            col = np.ceil(len(imgs) / row).astyle("uint8")

        for i, img in enumerate(imgs):
            plt.subplot(row, col, i + 1)
            plt.imshow(img, cmap=cmaps[i])

        plt.suptitle(title)
        plt.show()

    except ValueError:
        print("len(imgs) must be equal to the len of cmaps")


def rgb2gray_m(img:np.ndarray, method:str = "NTSC")->np.ndarray:
    """��RGBͼ��ת��grayͼ��

    Args:
        img (np.ndarray): ����ͼ������ΪRGBͼ��
        method (str, optional): ת��ģʽ. Defaults to "NTSC".

    Returns:
        np.ndarray: ���ͼ������Ϊgray-scaleͼ��
    """
