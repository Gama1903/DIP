from re import I
import cv2 as cv
import numpy as np

__all__ = [
    "getNegative",
    "logTrans",
    "gammaTrans",
]


def getNegative(img: np.ndarray, L: int = 256) -> np.ndarray:
    """图像反转，获得底片

    Args:
        img (np.ndarray): 输入图像
        L (int, optional): 灰度级数. Defaults to 256.

    Returns:
        np.ndarray: 输出图像，图像底片
    """

    return L - 1 - img


def logTrans(img: np.ndarray, c: int = 1) -> np.ndarray:
    """对数变换

    Args:
        img (np.ndarray): 输入图像
        c (int, optional): 系数. Defaults to 1.

    Returns:
        np.ndarray: 输出图像
    """
    maxVal = np.max(img)

    return (c * maxVal * np.log(1 + np.divide(img, maxVal))).astype("uint8")


def gammaTrans(img: np.ndarray, gamma: float, c: int = 1) -> np.ndarray:
    """伽马变换

    Args:
        img (np.ndarray): 输入图像
        gamma (float): 大于1灰度压缩，小于1灰度扩展
        c (int, optional): 系数. Defaults to 1.

    Returns:
        np.ndarray: 输出图像
    """
    maxVal = np.max(img)

    return (c * maxVal * np.divide(img, maxVal)**gamma).astype("uint8")


def main():
    flag_test = 2
    # getNegative()的测试程序
    if flag_test == 0:
        img_orig = cv.imread("images\\breast_digital_Xray.tif", flags=0)
        img_negative = getNegative(img_orig)
        cv.imshow("Original Image", img_orig)
        cv.imshow("Image Negative", img_negative)
    # logtrans()测试程序
    elif flag_test == 1:
        img_orig = cv.imread("images\\lena_rgb_orig.png", flags=0)
        img_processed = logTrans(img_orig)
        cv.imshow("Original Image", img_orig)
        cv.imshow("processed Image", img_processed)
    # gammatrans()的测试程序
    elif flag_test == 2:
        img_orig_1 = cv.imread("images\\fractured_spine.tif", flags=0)
        img_orig_2 = cv.imread("images\\washed_out_aerial_image.tif", flags=0)
        img_processed_1 = gammaTrans(img_orig_1, 0.3)
        img_processed_2 = gammaTrans(img_orig_2, 4)
        cv.imshow("Original Image 1", img_orig_1)
        cv.imshow("Processed Image 1", img_processed_1)
        cv.imshow("Original Image 2", img_orig_2)
        cv.imshow("Processed Image 2", img_processed_2)

    cv.waitKey()
    cv.destroyWindow()


if __name__ == "__main__":
    main()