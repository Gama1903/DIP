import cv2 as cv
import numpy as np
import imageio.v3 as iio
from typing import Callable

from repo.pkg.A_show_rgb2gray import imshow_m

__all__ = [
    "dec_pooling",
    "pooling",
]

# 最大池化函数
max_m = np.max
# 均值池化函数
average_m = lambda arr: int(np.average(arr))


def dec_pooling(pooling: Callable):
    """pooling函数的装饰器

    """
    def wrapper_func(InputMat: np.ndarray,
                     size: int,
                     step: int,
                     func: Callable = max_m) -> np.ndarray:

        if func == max_m:
            return pooling(InputMat, size, step, max_m)
        elif func == average_m:
            return pooling(InputMat, size, step, average_m)
        else:
            raise ValueError(
                "'{0}' is not a valid value; supported values are max_m and average_m. Defaults to max_m"
                .format(func))

    return wrapper_func


@dec_pooling
def pooling(InputMat: np.ndarray,
            size: int,
            step: int,
            func: Callable = max_m) -> np.ndarray:
    """池化

    Args:
        InputMat (np.ndarray): 输入矩阵, 必须为方形矩阵
        size (int): 池化块的尺寸
        step (int): 池化块的移动步长
        func (Callable, optional): 池化方法. max_m为最大池化, average_m为均值池化. Defaults to max_m

    Raises:
        ValueError: 输入矩阵必须为方形矩阵

    Returns:
        np.ndarray: 输出矩阵
    """
    if InputMat.shape[0] != InputMat.shape[1]:
        raise ValueError("InputMat must be a square matrix")

    InputMat = np.array(InputMat)
    # 计算输出矩阵的形状
    OutputMat_shape = (InputMat.shape[0] - size) // step + 1
    # 初始化输出矩阵
    OutputMat = np.zeros((OutputMat_shape, OutputMat_shape))

    i = 0
    j = 0
    out_i = 0
    out_j = 0
    while i < InputMat.shape[0]:
        if i + size > InputMat.shape[0]:
            break
        while j < InputMat.shape[1]:
            if j + size > InputMat.shape[1]:
                break
            OutputMat[out_i][out_j] = func(InputMat[i:i + size, j:j + size])
            j += step
            out_j += 1
        i += step
        out_i += 1
        j = 0
        out_j = 0

    return OutputMat.astype("uint8")


def main():
    # 测试程序
    img_orig = iio.imread("images\cameraman_orig.tif")
    img_pooled = pooling(img_orig, 3, 3)
    imshow_m((img_orig, img_pooled), ("Original Image", "Pooled Image"),
             ("gray", "gray"), 1, 2)

    # # opencv等效程序
    # img_orig = cv.imread("images\cameraman_orig.tif", flags=0)
    # img_pooled = pooling(img_orig, 3, 3)
    # cv.imshow("Original Image", img_orig)
    # cv.imshow("Pooled Image", img_pooled)
    # cv.waitKey()
    # cv.destroyWindow()


if __name__ == "__main__":
    main()
