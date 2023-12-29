import cv2
import numpy as np
import pywt
from PyQt5.QtGui import QImage, QPixmap


def cv2qt(cv_img):
    """ 把cv2格式的图像转换成QImage格式的图像 """
    height, width, channel = cv_img.shape
    bytes_per_line = channel * width
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return q_img  # 返回QImage格式的图像


def qt2cv(qt_img):
    """ 把QImage格式的图像转换成cv2格式的图像 """
    qt_img = qt_img.toImage()
    width = qt_img.width()
    height = qt_img.height()
    channel = qt_img.depth() // 8
    bytes_per_line = qt_img.bytesPerLine()
    ptr = qt_img.bits()
    ptr.setsize(height * bytes_per_line)
    arr = np.array(ptr).reshape(height, width, channel)
    # return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return arr


def histogram_equalization(channel):
    """
    :param img:  np array
    :return:  np array
    """
    # 1. 计算直方图
    hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])

    # 2. 计算累积直方图
    cum_hist = hist.cumsum()

    # 3. 归一化累积直方图
    cum_hist_normalized = (cum_hist - cum_hist.min()) * 255 / (cum_hist.max() - cum_hist.min())
    cum_hist_normalized = cum_hist_normalized.astype('uint8')

    # 4. 应用映射
    equalized_img = cum_hist_normalized[channel]

    return equalized_img


def histogram_equalization_qimage(qt_img):
    print("in histogram_equalization_color_qimage")
    cv_img = qt2cv(qt_img)

    print("cv_img.shape:", cv_img.shape)

    # 将图像转换到 YUV 色彩空间
    yuv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)

    # 对 Y 通道（亮度）应用直方图均衡化
    yuv_img[:, :, 0] = histogram_equalization(yuv_img[:, :, 0])

    print("yuv_img.shape:", yuv_img.shape)

    # 将图像转换回 BGR 色彩空间
    equalized_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB)

    # 转换为qimage
    return cv2qt(equalized_img)


# 2. 频域滤波
def frequency_domain_filter(image, filter_type='lowpass', cutoff_frequency=20):
    """
    :param image: np array（可以是灰度图或彩色图）
    :param filter_type: str, 'lowpass' or 'highpass'
    :param cutoff_frequency: int, 控制滤波器的大小
    :return: np array
    """

    def apply_filter(channel):
        # 计算傅里叶变换
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 创建掩膜
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # 应用掩膜
        if filter_type == 'lowpass':
            mask = np.zeros((rows, cols, 2), np.uint8)
            mask[crow - cutoff_frequency:crow + cutoff_frequency,
            ccol - cutoff_frequency:ccol + cutoff_frequency] = 1
        elif filter_type == 'highpass':
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[crow - cutoff_frequency:crow + cutoff_frequency,
            ccol - cutoff_frequency:ccol + cutoff_frequency] = 0
        else:
            raise ValueError("Unsupported filter type")

        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # 归一化
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return np.uint8(img_back)

    # 检查图像是否为彩色图像
    if len(image.shape) == 3:
        # 分别对每个通道应用滤波
        channels = cv2.split(image)
        filtered_channels = [apply_filter(channel) for channel in channels]
        # 合并通道
        filtered_image = cv2.merge(filtered_channels)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    else:
        # 灰度图直接应用滤波
        filtered_image = apply_filter(image)

    return filtered_image


def frequency_domain_filter_qimage(qt_img, filter_type='lowpass', cutoff_frequency=20):

    cv_img = qt2cv(qt_img)

    filtered_img = frequency_domain_filter(cv_img, filter_type, cutoff_frequency)

    return cv2qt(filtered_img)


# 3. 空域滤波
def custom_median_filter(channel, kernel_size):
    """
    中值滤波器
    :param channel:
    :param kernel_size:
    :return:
    """
    padded_img = np.pad(channel, kernel_size // 2, mode='constant')
    filtered_img = np.zeros_like(channel)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size]
            filtered_img[i, j] = np.median(neighborhood)

    return filtered_img


def custom_mean_filter(channel, kernel_size):
    """
    均值滤波器
    :param channel:
    :param kernel_size:
    :return:
    """
    padded_img = np.pad(channel, kernel_size // 2, mode='constant')
    filtered_img = np.zeros_like(channel)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size]
            filtered_img[i, j] = np.mean(neighborhood)

    return filtered_img


def custom_gaussian_filter(channel, kernel_size, sigma):
    """
    高斯滤波器
    :param channel:
    :param kernel_size:
    :param sigma:
    :return:
    """
    def gaussian_kernel(l=5, sig=1):
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    kernel = gaussian_kernel(kernel_size, sigma)
    padded_img = np.pad(channel, kernel_size // 2, mode='constant')
    filtered_img = np.zeros_like(channel)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size]
            filtered_img[i, j] = np.sum(neighborhood * kernel)

    return filtered_img


def filter_channel(channel, kernel, filter_type):
    if filter_type == 'median':
        # 使用自定义中值滤波器
        return custom_median_filter(channel, kernel)
    elif filter_type == 'mean':
        # 使用自定义均值滤波器
        return custom_mean_filter(channel, kernel)
    elif filter_type == 'gaussian':
        # 使用自定义高斯滤波器
        return custom_gaussian_filter(channel, kernel, 1)
    else:
        raise ValueError("Unsupported filter type")


def spatial_filter(image, kernel_size=5, filter_type='median'):
    # 检查图像是否为彩色图像
    if len(image.shape) == 3:
        # 分别对每个通道应用滤波
        channels = cv2.split(image)

        filtered_channels = [filter_channel(channel, kernel_size, filter_type) for channel in channels]
        # 合并通道
        filtered_image = cv2.merge(filtered_channels)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    else:
        # 灰度图直接应用滤波
        filtered_image = filter_channel(image, kernel_size, filter_type)

    return filtered_image


def spatial_domain_filtering_qimage(qt_img, filter_type='median', kernel_size=5):
    cv_img = qt2cv(qt_img)
    filtered_img = spatial_filter(cv_img, kernel_size, filter_type)
    return cv2qt(filtered_img)


# 4. 图像去噪
def filter_channel_denoise(channel, kernel_size, filter_type, sigma=None):
    pad_size = kernel_size // 2
    padded_channel = np.pad(channel, pad_size, mode='constant', constant_values=0)
    denoised_channel = np.zeros_like(channel)

    if filter_type == 'median':
        for i in range(pad_size, padded_channel.shape[0] - pad_size):
            for j in range(pad_size, padded_channel.shape[1] - pad_size):
                window = padded_channel[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                denoised_channel[i - pad_size, j - pad_size] = np.median(window)
    elif filter_type == 'mean':
        for i in range(pad_size, padded_channel.shape[0] - pad_size):
            for j in range(pad_size, padded_channel.shape[1] - pad_size):
                window = padded_channel[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]   # 把窗口中的像素值取出来
                denoised_channel[i - pad_size, j - pad_size] = np.mean(window)  # 计算均值
    elif filter_type == 'gaussian' and sigma is not None:

        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))  # 高斯核：二维高斯分布
        kernel /= np.sum(kernel)

        for i in range(pad_size, padded_channel.shape[0] - pad_size):
            for j in range(pad_size, padded_channel.shape[1] - pad_size):
                window = padded_channel[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                denoised_channel[i - pad_size, j - pad_size] = np.sum(window * kernel)
    else:
        raise ValueError("Unsupported filter type")

    return denoised_channel


def denoise_image(image, kernel_size, filter_type, sigma=None):
    if len(image.shape) == 3:
        channels = cv2.split(image)
        denoised_channels = [filter_channel_denoise(channel, kernel_size, filter_type, sigma) for channel in channels]
        denoised_image = cv2.merge(denoised_channels)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    else:
        denoised_image = filter_channel_denoise(image, kernel_size, filter_type, sigma)

    return denoised_image


def denoising_qimage(qt_img, filter_type='median', kernel_size=5):
    cv_img = qt2cv(qt_img)


    if filter_type in ['median', 'mean']:
        denoised_img = denoise_image(cv_img, kernel_size, filter_type)
    elif filter_type == 'gaussian':
        denoised_img = denoise_image(cv_img, kernel_size, filter_type, 1)
    else:
        raise ValueError("Unsupported filter type")

    return cv2qt(denoised_img)


#
# # 小波变换

def wavelet_transform_channel(channel):
    # 应用小波变换
    coeffs = pywt.dwt2(channel, 'haar')
    # 重构图像
    reconstructed = pywt.idwt2(coeffs, 'haar')
    return reconstructed


def wavelet_transform_image(cv_img):
    # 分离图像通道
    channels = cv2.split(cv_img)
    # 对每个通道应用小波变换
    transformed_channels = [wavelet_transform_channel(channel) for channel in channels]
    # 重构彩色图像
    transformed_img = cv2.merge(transformed_channels)
    # 归一化图像
    transformed_img = cv2.normalize(transformed_img, None, 0, 255, cv2.NORM_MINMAX)

    img = np.uint8(transformed_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def wavelet_transform_qimage(qt_img):
    """ Apply wavelet transform to a QImage """
    cv_img = qt2cv(qt_img)

    # 应用小波变换于彩色图像
    transformed_img = wavelet_transform_image(cv_img)

    # 将处理后的图像转换回QImage
    return cv2qt(transformed_img)


# 图像分割：canny，watershed，grabcut
def image_segmentation_qimage(qt_img, method='canny'):
    cv_img = qt2cv(qt_img)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if method == 'canny':
        edges = cv2.Canny(gray, 100, 200)
    elif method == 'watershed':
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  #
        kernel = np.ones((3, 3), np.uint8)  # 3x3的卷积核
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5) # 距离变换
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)    # 前景区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv_img, markers)
        cv_img[markers == -1] = [255, 0, 0]
        edges = cv_img
    elif method == 'grabcut':
        mask = np.zeros(cv_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (50, 50, 450, 290)
        cv2.grabCut(cv_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        cv_img = cv_img * mask2[:, :, np.newaxis]
        edges = cv_img
    else:
        raise ValueError("Unsupported segmentation method")

    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)


    return cv2qt(edges)


# 镜像
def mirror_qimage(qt_img, method='horizontal'):
    cv_img = qt2cv(qt_img)

    if method == 'horizontal':
        mirrored_img = cv2.flip(cv_img, 1)
    elif method == 'vertical':
        mirrored_img = cv2.flip(cv_img, 0)
    elif method == 'both':
        mirrored_img = cv2.flip(cv_img, -1)
    else:
        raise ValueError("Unsupported mirror method")

    # BGR
    mirrored_img = cv2.cvtColor(mirrored_img, cv2.COLOR_BGR2RGB)
    return cv2qt(mirrored_img)

