from email.mime import image
import cv2 as cv
import numpy as np


def pad_m(img: np.ndarray, pad_width: list, mode: str = "zero") -> np.ndarray:
    OutputImg = np.zeros(
        (img.shape[0] + pad_width[0] * 2, img.shape[1] + pad_width[1] * 2))
    OutputImg[pad_width[0]:pad_width[0] + img.shape[0],
              pad_width[1]:pad_width[1] + img.shape[1]] = img
    if mode == "zero":
        return OutputImg




def meanFilt_m(input_image, size_kernal):
    """Max Filter of Me

    Args:
        input_image (InputArray): Original Image
        size_kernal (int): size of kernal, 
                        must be odd and greater than or equal to 3
    Returns:
        output_image (OutputArray): Filtered Image
    """
    input_image_cp = np.copy(input_image)  # 原图副本

    kernal = np.ones((size_kernal, size_kernal))  # 初始化核

    num_pad = int((size_kernal - 1) / 2)  # 需填充的尺寸

    input_image_cp = pad_m(input_image, (size_kernal, size_kernal))  # 填充输入图像

    m, n = input_image_cp.shape  # 填充后图像尺寸

    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(num_pad, m - num_pad):
        for j in range(num_pad, n - num_pad):
            output_image[i, j] = np.sum(
                kernal *
                input_image_cp[i - num_pad:i + num_pad + 1,
                               j - num_pad:j + num_pad + 1]) / (size_kernal**2)

    output_image = output_image[num_pad:m - num_pad, num_pad:n - num_pad]  # 裁剪

    return output_image


def main():
    size_kernal = 11
    src = cv.cvtColor(
        cv.imread(cv.samples.findFile("Images\\blurring_orig.tif")),
        cv.COLOR_BGR2GRAY)
    dst = meanFilt_m(src, size_kernal)

    cv.imshow("Original Image", src)
    cv.imshow("Filtered Image", dst)
    cv.waitKey()
