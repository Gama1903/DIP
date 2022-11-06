import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from C_intensityTrans import logTrans
from A_show_rgb2gray import imshow_m

__all__ = [
    "fft2_cv",
]


def fft2_cv(img: np.ndarray) -> np.ndarray:
    """图像快速傅里叶变换的opencv实现

    Args:
        img (np.ndarray): 输入图像

    Returns:
        np.ndarray: 共3个输出, 分别为图像的傅里叶变换, 幅度谱, 相位谱
    """
    # 将图像从uint8转为float32
    img_f32 = np.float32(img)
    # 获得图像的行数和列数
    row, col = img.shape[:2]
    # 零填充（避免混叠）
    img_pad_1 = np.zeros((row * 2, col * 2), np.float32)
    img_pad_1[:row, :col] = img_f32
    # 最优DFT扩充尺寸
    rPad = cv.getOptimalDFTSize(img_pad_1.shape[0])
    cPad = cv.getOptimalDFTSize(img_pad_1.shape[1])
    # 零填充（提高fft算法效率）
    if rPad != img_pad_1.shape[0] and cPad != img_pad_1.shape[1]:
        img_pad_2 = np.zeros((rPad, cPad, 2), np.float32)
        img_pad_2[:img_pad_1.shape[0], :img_pad_1.shape[1], 0] = img_pad_1
    else:
        img_pad_2 = img_pad_1
    # 快速傅里叶变换
    img_fft = cv.dft(img_pad_2, flags=cv.DFT_COMPLEX_OUTPUT)
    # 中心化
    img_fft = np.fft.fftshift(img_fft)
    # 幅度谱
    img_amp = cv.magnitude(img_fft[:, :, 0], img_fft[:, :, 1])
    # 对数变换
    img_amp = logTrans(img_amp)
    # 相位谱
    img_phase = np.arctan2(img_fft[:, :, 1], img_fft[:, :, 0]) / np.pi * 180

    return img_fft, img_amp, img_phase


# 测试程序
def main():
    # fft_cv()
    img = cv.imread("images\\blown_ic.tif", flags=0)
    img_fft, img_amp, img_phase = fft2_cv(img)
    imshow_m((img_amp, img_phase), ("Amplitude Spectrum", "Phase Spectrum"),
             ("gray", "gray"), 1, 2)


if __name__ == "__main__":
    main()
