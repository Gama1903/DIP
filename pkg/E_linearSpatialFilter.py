import cv2 as cv
import numpy as np


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

    input_image_cp = np.pad(input_image_cp, (num_pad, num_pad),
                            mode="constant",
                            constant_values=0)  # 填充输入图像

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
