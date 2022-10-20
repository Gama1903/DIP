import imageio
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["imread_m, imshow_m,"]


def imread_m(path: str) -> np.ndarray:
    """通过imageio.imread()读入图片并转为np.ndarray

        Args:
            path (str): 源图片所在位置

        Returns:
            np.ndarray: 返回np.ndarray，元素类型uint8
        """
    return np.array(imageio.imread(path))


def imshow_m(title: str, imgs: tuple, cmaps: list, row: int = 0, col: int = 0):
    """展示图片

    Args:
        title (str): 图像标题
        imgs (tuple): 图片元组
        cmaps (list): plt以何种图片类型展示图片
        row (int, optional): 指令row. Defaults to 0.
        col (int, optional): 指令col. Defaults to 0.
    """
    try:
        if row == 0 and col != 0:
            row = np.ceil(len(imgs) / col).astyle("uint8")
        elif row != 0 and col == 0:
            col = np.ceil(len(imgs) / row).astyle("uint8")
        elif row * col < len(imgs):
            # 尽量以方正的形式去展示图片
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
    """将RGB图像转成gray图像

    Args:
        img (np.ndarray): 输入图像，类型为RGB图像
        method (str, optional): 转换模式. Defaults to "NTSC".

    Returns:
        np.ndarray: 输出图像，类型为gray-scale图像
    """
