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

    return L - 1 - np.copy(img)


def logTrans(img: np.ndarray) -> np.ndarray:
    """对数变换

    Args:
        img (np.ndarray): 输入图像

    Returns:
        np.ndarray: 输出图像
    """

    c = 255 // np.log(1 + np.abs(np.max(img)))
    a = np.log(1 + np.copy(img))

    return 1 * np.log(1 + np.copy(img))


def gammaTrans(img: np.ndarray, gamma: float) -> np.ndarray:
    """伽马变换

    Args:
        img (np.ndarray): 输入图像
        gamma (float): 大于1灰度压缩，小于1灰度扩展

    Returns:
        np.ndarray: 输出图像
    """

    img = np.power(np.copy(img), gamma)
    c = 255 // np.log(1 + np.abs(np.max(img)))

    return 1 * img


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
        img_orig = cv.imread("images\\DFT_no_log.tif")
        img_processed = logTrans(img_orig)
        cv.imshow("Original Image", img_orig)
        # cv.imshow("processed Image", img_processed)
    # gammatrans()的测试程序
    elif flag_test == 2:
        img_orig = cv.imread("images\\washed_out_aerial_image.tif", flags=0)
        img_processed = gammaTrans(img_orig, 1.5)
        cv.imshow("Original Image", img_orig)
        cv.imshow("processed Image", img_processed)

    cv.waitKey()
    cv.destroyWindow()


if __name__ == "__main__":
    main()