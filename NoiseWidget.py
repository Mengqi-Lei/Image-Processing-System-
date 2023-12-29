import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QPushButton
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from img_process_algorithms import cv2qt, qt2cv

import warnings

warnings.filterwarnings("ignore")


class AddNoiseWidget(QDialog):

    def __init__(self, img=None):
        super(AddNoiseWidget, self).__init__()
        self.setWindowTitle('添加噪声')
        self.resize(800, 600)
        # 固定对话框大小
        self.setFixedSize(self.width(), self.height())

        # 添加成员：
        self.origin_image = img
        self.added_noise_image = self.origin_image

        self.initUI()

        # print("AddNoiseWidget init finished")

        self.display_image(self.origin_image)

        # print("AddNoiseWidget display finished")

    def initUI(self):
        # 设置字体
        font = QFont("楷体", 15)

        # 总体布局（垂直）
        layout = QVBoxLayout()

        # 图片展示区域（上半部分）
        self.image_label = QLabel("在这里展示图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel{border: 3px solid channel;}")

        self.image_label.setFont(font)
        layout.addWidget(self.image_label)

        # 控制区域（下半部分）
        control_layout = QHBoxLayout()

        # 创建控件
        self.noise_type_combobox = QComboBox()
        self.noise_type_combobox.addItems(["高斯噪声", "椒盐噪声", "泊松噪声", "均匀噪声"])  # 添加噪声类型选项
        self.noise_type_combobox.setFont(font)

        control_layout.addWidget(self.noise_type_combobox)

        self.noise_intensity_label = QLabel("选择噪声强度")
        self.noise_intensity_label.setFont(font)
        control_layout.addWidget(self.noise_intensity_label)

        self.noise_intensity_slider = QSlider(Qt.Horizontal)
        self.noise_intensity_slider.setMinimum(0)
        self.noise_intensity_slider.setMaximum(100)  # 假设噪声强度的范围是 0 到 100
        self.noise_intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_intensity_slider.setTickInterval(10)
        self.noise_intensity_slider.setFont(font)

        control_layout.addWidget(self.noise_intensity_slider)

        # 应用
        self.apply_button = QPushButton("应用")
        self.apply_button.setFont(font)
        self.apply_button.clicked.connect(self.update_noise_image)
        control_layout.addWidget(self.apply_button)

        self.confirm_button = QPushButton("确认")
        self.confirm_button.setFont(font)
        self.confirm_button.clicked.connect(self.confirm_noise_addition)  # 需要定义相应的槽函数
        control_layout.addWidget(self.confirm_button)

        # 将控制布局添加到总体布局
        layout.addLayout(control_layout)

        # 设置对话框的主布局
        self.setLayout(layout)

    # # 定义重绘事件：每次重绘都自动显示原图
    # def paintEvent(self, event):
    #     self.display_image(self.image_processed)

    def display_image(self, image):
        """
        显示图片
        :param img:   是一个 QPixmap 对象
        :return:
        """
        # 深拷贝


        if image is None:
            return
        # pixmap = img
        # pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        # self.image_label.setPixmap(pixmap)
        # print("display image finished")
        # self.update()

        img = image.copy()
        # 如果是Qimage对象
        if isinstance(img, QImage):
            pixmap = QPixmap.fromImage(img)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            print("display image finished")
            self.update()
        # 如果是QPixmap对象
        elif isinstance(img, QPixmap):
            pixmap = img
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            print("display image finished")
            self.update()
        else:
            raise ValueError("Unsupported image type")

    def update_noise_image(self):
        # 获取噪声类型和强度
        noise_type = self.noise_type_combobox.currentText()
        intensity = self.noise_intensity_slider.value()

        # 将 QPixmap 转换为 cv2 格式
        cv_img = qt2cv(self.origin_image)

        print("qt2cv finished")

        # 高斯噪声
        if noise_type == "高斯噪声":
            sigma = intensity * 0.1  # 噪声强度调整因子
            gauss = np.random.normal(0, sigma, cv_img.shape).reshape(cv_img.shape)
            noisy_img = cv_img + gauss

        # 椒盐噪声
        elif noise_type == "椒盐噪声":
            # 计算需要添加的噪声点的数量
            # intensity 的范围假设为 0 到 100
            # 0 表示没有噪声，100 表示所有像素点都是噪声
            # 假设噪声点的数量为 img_size * intensity / 100
            img_size = cv_img.shape[0] * cv_img.shape[1]
            num_noise = int(img_size * intensity / 100)
            # 生成噪声点的坐标
            x = np.random.randint(0, cv_img.shape[0], num_noise)
            y = np.random.randint(0, cv_img.shape[1], num_noise)
            # 将噪声点的像素值设置为 0 或 255，注意三个通道都要设置
            for i in range(num_noise):
                cv_img[x[i], y[i], :] = 255 if np.random.randint(0, 2) == 0 else 0


            noisy_img = cv_img



        # 泊松噪声
        elif noise_type == "泊松噪声":
            vals = len(np.unique(cv_img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy_img = np.random.poisson(cv_img * vals) / float(vals)

        # 均匀噪声
        elif noise_type == "均匀噪声":
            uniform_noise = np.random.uniform(-intensity, intensity, cv_img.shape)
            noisy_img = cv_img + uniform_noise

        else:
            raise ValueError("Unsupported noise type")

        print("noise type finished")

        # 限制像素值在合理范围内
        noisy_img = np.clip(noisy_img, 0, 255)
        # 转换为bgr
        noisy_img = cv2.cvtColor(noisy_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # 转换回 QImage 格式以显示
        noisy_qimage = cv2qt(noisy_img.astype(np.uint8))
        print("type of noisy_qimage: ", type(noisy_qimage))

        # 保存到成员变量中
        self.added_noise_image = noisy_qimage
        # 显示图片
        self.display_image(self.added_noise_image)


    def confirm_noise_addition(self):
        # 发送信号
        # self.image_processed.emit("ok")
        # 关闭对话框
        self.close()

    def load_image(self, path):
        # 读取图片
        img = QImage(path)
        # 转化为pixmap格式
        pixmap = QPixmap.fromImage(img)
        # 保存到成员变量中
        self.origin_image = pixmap
        print("load image finished" + path)
        # 显示图片
        self.display_image(pixmap)


# 测试对话框
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    demo = AddNoiseWidget()
    demo.show()
    sys.exit(app.exec_())
