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
    input_image_cp = np.copy(input_image)  # Ô­Í¼¸±±¾

    kernal = np.ones((size_kernal, size_kernal))  # ³õÊ¼»¯ºË

    num_pad = int((size_kernal - 1) / 2)  # ĞèÌî³äµÄ³ß´ç

    input_image_cp = np.pad(input_image_cp, (num_pad, num_pad),
                            mode="constant",
                            constant_values=0)  # Ìî³äÊäÈëÍ¼Ïñ

    m, n = input_image_cp.shape  # Ìî³äºóÍ¼Ïñ³ß´ç

    output_image = np.copy(input_image_cp)  # Êä³öÍ¼Ïñ

    # ¿Õ¼äÂË²¨
    for i in range(num_pad, m - num_pad):
        for j in range(num_pad, n - num_pad):
            output_image[i, j] = np.sum(
                kernal *
                input_image_cp[i - num_pad:i + num_pad + 1,
                               j - num_pad:j + num_pad + 1]) / (size_kernal**2)

    output_image = output_image[num_pad:m - num_pad, num_pad:n - num_pad]  # ²Ã¼ô

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
