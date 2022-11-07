import cv2 as cv
import numpy as np
from A_show_rgb2gray import imshow_m

__all__ = [
    "equalizeHist_m",
]


def equalizeHist_m(InputImg: np.ndarray) -> np.ndarray:
    """直方图均衡化

    Args:
        InputImg (np.ndarray): 输入图像

    Returns:
        np.ndarray: 输出图像
    """
    # 输出图像初始化
    OutputImg = np.copy(InputImg)
    # 原图的副本
    InputImg_cp = np.copy(InputImg)
    # 原图的尺寸
    row, col = InputImg_cp.shape
    # 原图的像素总数
    num_pixel = row * col
    # 概率密度函数, probability density function
    pdf = []
    for i in range(256):
        pdf.append(np.sum(InputImg_cp == i) / num_pixel)
    # 累积分布函数, cumulation distribution function
    cdf = 0
    for i in range(256):
        cdf = cdf + pdf[i]
        # 求解输出图像
        OutputImg[np.where(InputImg_cp == i)] = 255 * cdf
    return OutputImg


# 测试程序
def main():
    flag_test = 1

    # 主测试程序
    if flag_test == 0:
        img_orig = cv.imread("images\washed_out_aerial_image.tif", flags=0)
        img_processed = equalizeHist_m(img_orig)
        imshow_m((img_orig, img_processed),
                 ("Original Image", "Processed Image"), ("gray", "gray"), 1, 2)
    # opencv等效程序
    elif flag_test == 1:
        img_orig = cv.imread("images\washed_out_aerial_image.tif", flags=0)
        img_processed = cv.equalizeHist(img_orig)
        imshow_m((img_orig, img_processed),
                 ("Original Image", "Processed Image"), ("gray", "gray"), 1, 2)


if __name__ == "__main__":
    main()
