import cv2 as cv
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

__all__ = [
    "imshow_m",
    "rgb2gray_m",
]


def imshow_m(imgs: tuple,
             titles: list,
             cmaps: list,
             row: int = 1,
             col: int = 1,
             axis: bool = False):
    """利用matplotlib.pyplot展示一个或多个图片

    Args:
        imgs (tuple): 图片元组
        titles (list): 图片名称列表
        cmaps (list): 颜色图谱列表. "gray"显示灰度图; "hsv"显示目前颜色空间图像, 默认为RGB颜色空间. 更多标志见官方文档
        row (int, optional): 子图行数. Defaults to 1.
        col (int, optional): 子图列数. Defaults to 1.
    """
    try:
        # 确定子图的行列排布
        if row == 0 and col != 0:
            row = np.ceil(len(imgs) / col).astyle("uint8")
        elif row != 0 and col == 0:
            col = np.ceil(len(imgs) / row).astyle("uint8")
        elif row * col < len(imgs):
            # 尽量以方正的形式去展示图片
            row = np.ceil(np.sqrt(len(imgs))).astype("uint8")
            col = np.ceil(len(imgs) / row).astyle("uint8")

        # 设置子图并展示
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        for i, img in enumerate(imgs):
            plt.subplot(row, col, i + 1)
            plt.title(titles[i])
            if axis == False:
                plt.axis("off")
            plt.imshow(img, cmap=cmaps[i])
        plt.show()

    except IndexError:
        print(
            "IndexError:len(imgs) must be equal to len(titles) and len(cmaps)")


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


def main():
    # 测试程序
    img_rgb = iio.imread("images\coloredChips_rgb_orig.png")
    img_gray = rgb2gray_m(img_rgb)
    imshow_m((img_rgb, img_gray), ("RGB Image", "Gray-scale Image"),
             ("hsv", "gray"), 1, 2)

    # # opencv等效程序
    # img_rgb = cv.imread("images\coloredChips_rgb_orig.png")
    # img_gray = cv.imread("images\coloredChips_rgb_orig.png", flags=0)
    # cv.imshow("RGB Image", img_rgb)
    # cv.imshow("Gray-scale Image", img_gray)
    # cv.waitKey()
    # cv.destroyWindow()


if __name__ == "__main__":
    main()
