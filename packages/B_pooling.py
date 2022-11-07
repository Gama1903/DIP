import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from typing import Callable

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

    Args:
        pooling (Callable): pooling函数
    """
    def wrapper_func(InputMat: np.ndarray,
                     size: int,
                     func: Callable = max_m) -> np.ndarray:

        if func == max_m:
            return pooling(InputMat, size, max_m)
        elif func == average_m:
            return pooling(InputMat, size, average_m)
        else:
            raise ValueError(
                "'{0}' is not a valid value; supported values are max_m and average_m. Defaults to max_m"
                .format(func))

    return wrapper_func


@dec_pooling
def pooling(InputMat: np.ndarray,
            size: int,
            func: Callable = max_m) -> np.ndarray:
    """池化

    Args:
        InputMat (np.ndarray): 输入矩阵, 必须为方形矩阵
        size (int): 池化块的尺寸
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
    OutputMat_shape = InputMat.shape[0] // size
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
            j += size
            out_j += 1
        i += size
        out_i += 1
        j = 0
        out_j = 0

    return OutputMat.astype("uint8")


# 测试程序
def main():
    flag_test = 0

    ax1 = plt.subplot(121)
    img_orig = iio.imread("images\cameraman_orig.tif")
    ax1.set_title("Original Image")
    ax1.set_ylim(img_orig.shape[0], 0)
    ax1.set_xlim(0, img_orig.shape[1])
    ax1.set_axis_off()
    plt.imshow(img_orig, cmap="gray")

    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    # Maxpooling
    if flag_test == 0:
        img_pooled = pooling(img_orig, 3)
        ax2.set_title("Maxpooled Image")
    # Averagepooling
    elif flag_test == 1:
        img_pooled = pooling(img_orig, 3)
        ax2.set_title("Maxpooled Image")
    ax2.set_axis_off()
    plt.imshow(img_pooled, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
