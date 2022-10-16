import numpy as np


def equalizeHist_m(input_image):
    """equalizeHist() of me

    Args:
        input_image (InputArray): original image

    Returns:
        output_image (OutputArray): equalized image
    """
    output_image = np.copy(input_image) # 均衡化后的图像，初始化为原图

    input_image_cp = np.copy(input_image) # 原图的副本

    m, n = input_image_cp.shape # 原图的尺寸

    pixels_total_num = m*n # 原图的像素总数

    P_input_image_grayscale = [] # 原图各灰度级概率，即原图直方图

    # 求原图直方图
    for i in range(256):
        P_input_image_grayscale.append(np.sum(input_image_cp == i) / pixels_total_num)

    # 求解输出图像
    F = 0 # 原图的灰度级概率分布函数
    for i in range(256):
        F = F + P_input_image_grayscale[i]
        output_image[np.where(input_image_cp == i)] = 255 * F

    return output_image

