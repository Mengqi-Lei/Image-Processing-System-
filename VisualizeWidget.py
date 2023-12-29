import os
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib import pyplot as plt
# 中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from img_process_algorithms import cv2qt, qt2cv
import warnings
warnings.filterwarnings("ignore")




class VisualizeDataWidget(QDialog):
    """
    # 传入的图片为QImage对象
    """
    def __init__(self, img1=None, img2=None):
        super(VisualizeDataWidget, self).__init__()
        self.setWindowTitle('可视化数据')
        self.setWindowIcon(QIcon('icons/可视化.png'))
        # self.resize(1500, 800)
        self.initUI()

        self.img_origin = img1  # numpy形式的图像
        self.img_processed = img2
    def initUI(self):
        self.leftWidget = QWidget()
        self.rightWidget = QWidget()
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.leftWidget.setLayout(self.leftLayout)
        self.rightWidget.setLayout(self.rightLayout)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.leftWidget)
        self.splitter.addWidget(self.rightWidget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([500, 500])
        self.mainLayout = QHBoxLayout()
        self.mainLayout.addWidget(self.splitter)
        self.setLayout(self.mainLayout)

        # 添加标签：处理前图像、处理后图像、处理前图像频谱图、处理后图像频谱图、处理前图像直方图、处理后图像直方图
        self.label_before = QLabel('处理前图像')
        self.label_before.setAlignment(Qt.AlignCenter)
        self.label_before.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_before.setFixedSize(800, 60)
        self.leftLayout.addWidget(self.label_before)

        self.label_after = QLabel('处理后图像')
        self.label_after.setAlignment(Qt.AlignCenter)
        self.label_after.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_after.setFixedSize(800, 60)
        self.rightLayout.addWidget(self.label_after)



        self.label_origin = QLabel('原始图像')
        self.label_origin.setAlignment(Qt.AlignCenter)
        self.label_origin.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_origin.setFixedSize(800, 400)
        self.leftLayout.addWidget(self.label_origin)

        self.label_processed = QLabel('处理后图像')
        self.label_processed.setAlignment(Qt.AlignCenter)
        self.label_processed.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_processed.setFixedSize(800, 400)
        self.rightLayout.addWidget(self.label_processed)

        self.label_origin_spectrogram = QLabel('原始图像频谱图')
        self.label_origin_spectrogram.setAlignment(Qt.AlignCenter)
        self.label_origin_spectrogram.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_origin_spectrogram.setFixedSize(800, 400)
        self.leftLayout.addWidget(self.label_origin_spectrogram)

        self.label_processed_spectrogram = QLabel('处理后图像频谱图')
        self.label_processed_spectrogram.setAlignment(Qt.AlignCenter)
        self.label_processed_spectrogram.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_processed_spectrogram.setFixedSize(800, 400)
        self.rightLayout.addWidget(self.label_processed_spectrogram)

        self.label_origin_histogram = QLabel('原始图像直方图')
        self.label_origin_histogram.setAlignment(Qt.AlignCenter)
        self.label_origin_histogram.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_origin_histogram.setFixedSize(800, 400)
        self.leftLayout.addWidget(self.label_origin_histogram)

        self.label_processed_histogram = QLabel('处理后图像直方图')
        self.label_processed_histogram.setAlignment(Qt.AlignCenter)
        self.label_processed_histogram.setStyleSheet("QLabel{border: 3px solid channel;}")
        self.label_processed_histogram.setFixedSize(800, 400)
        self.rightLayout.addWidget(self.label_processed_histogram)



        #设置标签字体
        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        #设置为楷体
        font.setFamily('楷体')


        for i in range(self.leftLayout.count()):
            self.leftLayout.itemAt(i).widget().setFont(font)
            self.leftLayout.itemAt(i).widget().setAlignment(Qt.AlignCenter)
        for i in range(self.rightLayout.count()):
            self.rightLayout.itemAt(i).widget().setFont(font)
            self.rightLayout.itemAt(i).widget().setAlignment(Qt.AlignCenter)

    def generate_spectrogram(self, img_np):
        # 将图像转换为灰度（如果它是彩色的）
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # 计算FFT
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        # 将频谱图转换为可显示的格式
        spectrum_img = np.asarray(magnitude_spectrum, dtype=np.uint8)
        spectrum_img = cv2.normalize(spectrum_img, None, 0, 255, cv2.NORM_MINMAX)
        return spectrum_img

    def generate_histogram(self, img_np):
        plt.figure(figsize=(8,4))
        # 标题：H-S直方图
        plt.title("H-S直方图")
        # 用plt画直方图
        if len(img_np.shape) == 2:  # 灰度图像
            plt.hist(img_np.ravel(), 256, [0, 256])

        else:  # 彩色图像
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.legend(['blue', 'green', 'red'])

        # 保存
        plt.savefig('histogram.png')
        # 读取
        histogram = cv2.imread('histogram.png')
        # 删除
        os.remove('histogram.png')

        return histogram


    def display_data(self):
        # 将 QImage 转换为 NumPy 数组（OpenCV格式）
        img_origin_np = qt2cv(self.img_origin)
        img_processed_np = qt2cv(self.img_processed)

        print("img_origin_np.shape: ", img_origin_np.shape)
        print("img_processed_np.shape: ", img_processed_np.shape)

        # 生成并显示频谱图和直方图
        self.display_spectrogram_and_histogram(img_origin_np, is_original=True)
        self.display_spectrogram_and_histogram(img_processed_np, is_original=False)

        print("display finished")

    def display_spectrogram_and_histogram(self, img_np, is_original):
        img_np= cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        # 生成频谱图和直方图
        spectrogram = self.generate_spectrogram(img_np)
        histogram = self.generate_histogram(img_np)

        # 把spectrogram转换为多通道
        spectrogram = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)



        # 在label中显示图像
        if is_original:
            # self.label_origin.setPixmap(QPixmap.fromImage(cv2qt(img_np)))
            # self.label_origin_spectrogram.setPixmap(QPixmap.fromImage(cv2qt(spectrogram)))
            # self.label_origin_histogram.setPixmap(QPixmap.fromImage(cv2qt(histogram)))
            self.display_img_in_label(img_np, self.label_origin)
            self.display_img_in_label(spectrogram, self.label_origin_spectrogram)
            self.display_img_in_label(histogram, self.label_origin_histogram)

        else:
            # self.label_processed.setPixmap(QPixmap.fromImage(cv2qt(img_np)))
            # self.label_processed_spectrogram.setPixmap(QPixmap.fromImage(cv2qt(spectrogram)))
            # self.label_processed_histogram.setPixmap(QPixmap.fromImage(cv2qt(histogram)))
            self.display_img_in_label(img_np, self.label_processed)
            self.display_img_in_label(spectrogram, self.label_processed_spectrogram)
            self.display_img_in_label(histogram, self.label_processed_histogram)

        # self.label_origin.setPixmap(QPixmap.fromImage(cv2qt(img_np)))

    def display_img_in_label(self, img_np, label):
        # 需要将img_np转换为QImage
        img_qt = cv2qt(img_np)
        # 将图片等比缩放到label的大小
        img_qt = img_qt.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        # 在label中显示图片
        label.setPixmap(QPixmap.fromImage(img_qt))







if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = VisualizeDataWidget()
    main.show()
    sys.exit(app.exec_())
