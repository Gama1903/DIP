import cv2 as cv
import numpy as np
from typing import Tuple
from typing import Optional
from A_show_rgb2gray import imshow_m
from C_intensityTrans import logTrans

__all__ = [
    "dft",
    "create_filter",
    "fdf",
]


def dft(img_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """图像离散傅里叶变换的实现

    Args:
        img (np.ndarray): 输入图像

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 共3个输出，分别为图像的傅里叶变换复数谱、幅度谱、相位谱
    """
    # 将图像从uint8转为float32
    img_f32 = np.float32(img_input)
    # 获得图像的行数和列数
    row, col = img_input.shape[:2]
    # 零填充（避免混叠）
    rPad, cPad = 2 * row, 2 * col
    img_pad = np.zeros((rPad, cPad), np.float32)
    img_pad[:row, :col] = img_f32
    # 离散傅里叶变换和中心化
    img_dft = np.fft.fftshift(cv.dft(img_pad, flags=cv.DFT_COMPLEX_OUTPUT))
    # 幅度谱, amplitude spectrum
    img_amp = cv.magnitude(img_dft[:, :, 0], img_dft[:, :, 1])
    # 对数变换
    img_amp = logTrans(img_amp)
    # 相位谱, phase spectrum
    img_phase = np.arctan2(img_dft[:, :, 1], img_dft[:, :, 0]) / np.pi * 180

    return img_dft, img_amp, img_phase


def create_filter(img_dft: np.ndarray,
                  filter: str,
                  D0: int,
                  n: Optional[int] = None) -> np.ndarray:
    """构造滤波器

    Args:
        img_dft (np.ndarray): 频域图像
        filter (str): 滤波器类型
        D0 (int): 截止频率
        n (int, optional): 布特沃斯滤波器的阶数. Defaults to None.

    Raises:
        ValueError: 当filter等于"blpf"或"bhpf"时, n必须指定一个整数

    Returns:
        np.ndarray: 滤波器矩阵
    """
    if filter in ["blpf", "bhpf"]:
        if n is None:
            raise ValueError("当filter等于'blpf'或'bhpf'时, n必须指定一个整数")
        if not isinstance(n, int):
            raise ValueError("n必须为整数")

    # 频域图像尺寸
    row, col = img_dft[1].shape[:2]
    # 频域图像坐标
    u, v = np.ogrid[0:row:1, 0:col:1]
    # 频率矩阵
    D = np.hypot(u - row // 2, v - col // 2)

    # 定义滤波器映射关系
    filter_map = {
        # 布特沃斯低通滤波器, Butterworth low-pass filter
        "blpf": lambda D0, n: 1.0 / (1.0 + np.power(D / (D0 + 1e-8), 2 * n)),
        # 理想低通滤波器, ideal low-pass filter
        "ilpf": lambda D0, n: np.zeros((row, col), np.float32)
        if D <= D0 else 0,
        # 高斯低通滤波器，Gaussian low-pass filter
        "glpf": lambda D0, n: np.exp(-1 * np.power(D, 2) / 2 * D0),
        # 布特沃斯高通滤波器, Butterworth high-pass filter
        "bhpf": lambda D0, n: 1 - 1.0 /
        (1.0 + np.power(D / (D0 + 1e-8), 2 * n)),
        # 理想高通滤波器, ideal high-pass filter
        "ihpf": lambda D0, n: np.ones((row, col), np.float32)
        if D <= D0 else 1,
        # 高斯高通滤波器，Gaussian high-pass filter
        "ghpf": lambda D0, n: 1 - np.exp(-1 * np.power(D, 2) / 2 * D0)
    }

    # 根据滤波器类型查找滤波器函数
    filter_func = filter_map.get(filter)

    # 如果找不到滤波器类型，则抛出错误
    if filter_func is None:
        raise ValueError(f"未知的滤波器类型: {filter}")

    # 调用滤波器函数，并返回滤波器的输出
    return filter_func(D0, n)


def fdf(img_input: np.ndarray,
        filter: str,
        D0: int,
        n: int = None) -> np.ndarray:
    """频域滤波, frequency domain filtering
    Args:
        img_input (np.ndarray): 输入图像
        filter (str): 滤波器类型, 有效类型为, 低通滤波 "blpf", "ilpf", "glpf"; 高通滤波 "bhpf", "ihpf", "ghpf"
        D0 (int): 截止频率
        n (int, optional): 布特沃斯滤波器的阶数. Defaults to None.
    Raises:
        ValueError: 当filter等于"blpf"或"bhpf"时, n必须指定一个整数

    Returns:
        np.ndarray: 输出图像
    """

    # 离散傅里叶正变换
    img_dft = dft(img_input)

    # 构造滤波器
    kernel = create_filter(img_dft, filter, D0, n)

    # 频域滤波
    img_filtered = img_dft[1][:img_dft[1].shape[0], :img_dft[1].
                              shape[1]] * kernel

    # 离散傅里叶反变换和去中心化
    img_idft = np.fft.ifft2(img_filtered,
                            s=(img_input.shape[0], img_input.shape[1]))

    # 输出图像尺寸取原图尺寸
    img_output = img_idft[:img_input.shape[0], :img_input.shape[1]]

    # 输出规范为0-255
    img_output = cv.normalize(np.abs(img_output), None, 0, 255,
                              cv.NORM_MINMAX).astype("uint8")

    return img_output


# 测试程序
def main():
    flag_test = 1
    # dft_cv()
    if flag_test == 0:
        img_orig = cv.imread("images\cameraman_orig.tif", flags=0)
        img_dft = dft(img_orig)
        img_amp = img_dft[1]
        img_phase = img_dft[2]
        imshow_m((img_amp, img_phase),
                 ("Amplitude Spectrum", "Phase Spectrum"), ("gray", "gray"), 1,
                 2)

    # fdf()
    elif flag_test == 1:
        img_orig = cv.imread("images\cameraman_orig.tif", flags=0)
        img_filtered = fdf(img_orig, "blpf", 100, 2)
        imshow_m((img_orig, img_filtered),
                 ("Original Image", "Filtered Image"), ("gray", "gray"), 1, 2)


if __name__ == "__main__":
    main()
