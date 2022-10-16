import numpy as np


def equalizeHist_m(input_image):
    """equalizeHist() of me

    Args:
        input_image (InputArray): original image

    Returns:
        output_image (OutputArray): equalized image
    """
    output_image = np.copy(input_image) # ���⻯���ͼ�񣬳�ʼ��Ϊԭͼ

    input_image_cp = np.copy(input_image) # ԭͼ�ĸ���

    m, n = input_image_cp.shape # ԭͼ�ĳߴ�

    pixels_total_num = m*n # ԭͼ����������

    P_input_image_grayscale = [] # ԭͼ���Ҷȼ����ʣ���ԭͼֱ��ͼ

    # ��ԭͼֱ��ͼ
    for i in range(256):
        P_input_image_grayscale.append(np.sum(input_image_cp == i) / pixels_total_num)

    # ������ͼ��
    F = 0 # ԭͼ�ĻҶȼ����ʷֲ�����
    for i in range(256):
        F = F + P_input_image_grayscale[i]
        output_image[np.where(input_image_cp == i)] = 255 * F

    return output_image

