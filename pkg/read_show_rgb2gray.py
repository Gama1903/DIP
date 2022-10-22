import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

__all__ = [
    "imread_m",
    "imshow_m",
    "rgb2gray_m",
]


def imread_m(path: str) -> np.ndarray:
    """通过imageio.imread()读入图片并转为np.ndarray

        Args:
            path (str): 源图片所在位置

        Returns:
            np.ndarray: 输出np.ndarray
        """
    return np.array(imageio.imread(path))


def imshow_m(imgs: tuple,
             titles: list,
             cmaps: list,
             row: int = 1,
             col: int = 1):
    """利用matplotlib.pyplot展示图片

    Args:
        imgs (tuple): 图片元组
        titles (list): 图片名称列表
        cmaps (list): 颜色图谱列表. "gray"显示灰度图; "hsv"显示目前颜色空间图像, 默认为RGB颜色空间. 更多标志见官方文档
        row (int, optional): 子图行数. Defaults to 1.
        col (int, optional): 子图列数. Defaults to 1.
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

        plt.rcParams['font.sans-serif'] = ['KaiTi']
        for i, img in enumerate(imgs):
            plt.subplot(row, col, i + 1)
            plt.title(titles[i])
            plt.axis("off")
            plt.imshow(img, cmap=cmaps[i])

        plt.show()

    except IndexError:
        print("Error:'len(imgs) must be equal to len(titles) and len(cmaps)'")


def rgb2gray_m(InputImg: np.ndarray, method: str = "NTSC") -> np.ndarray:
    """将RGB图像转成gray图像

    Args:
        InputImg (np.ndarray): 输入图像，类型为RGB图像
        method (str, optional): 转换模式. Defaults to "NTSC".

    Returns:
        np.ndarray: 输出图像，类型为gray-scale图像
    """
    if method == "average":
        OutputImg = InputImg[:, :, 0] / 3 + InputImg[:, :,
                                                     1] / 3 + InputImg[:, :,
                                                                       2] / 3
    else:
        OutputImg = InputImg[:, :,
                             0] * 0.2989 + InputImg[:, :,
                                                    1] * 0.5870 + InputImg[:, :,
                                                                           2] * 0.1140
    return OutputImg


# 测试程序
def main():
    img1 = imread_m("repo\images\coloredChips_rgb_orig.png")
    img2 = rgb2gray_m(img1)
    imshow_m((img1, img2), ("coloredChips_rgb", "coloredChips_gray"),
             ("hsv", "gray"), 1, 2)


if __name__ == "__main__":
    main()
