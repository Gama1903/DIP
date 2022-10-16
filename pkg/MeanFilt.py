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
    input_image_cp = np.copy(input_image)  # ԭͼ����

    kernal = np.ones((size_kernal, size_kernal))  # ��ʼ����

    num_pad = int((size_kernal - 1) / 2)  # �����ĳߴ�

    input_image_cp = np.pad(input_image_cp, (num_pad, num_pad),
                            mode="constant",
                            constant_values=0)  # �������ͼ��

    m, n = input_image_cp.shape  # ����ͼ��ߴ�

    output_image = np.copy(input_image_cp)  # ���ͼ��

    # �ռ��˲�
    for i in range(num_pad, m - num_pad):
        for j in range(num_pad, n - num_pad):
            output_image[i, j] = np.sum(kernal*input_image_cp[i - num_pad:i + num_pad + 1,
                               j - num_pad:j + num_pad + 1]) / (size_kernal**2)

    output_image = output_image[num_pad:m - num_pad, num_pad:n - num_pad]  # �ü�

    return output_image
