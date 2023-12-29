import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                             QComboBox, QFileDialog, QDialog, QLineEdit, QSpacerItem, QSizePolicy, QCheckBox,
                             QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QTransform
from PyQt5.QtCore import Qt, QThread
from img_process_algorithms import *
from NoiseWidget import AddNoiseWidget
from VisualizeWidget import VisualizeDataWidget
sys.path.append('E:/Files/PythonProjects/ImageProcessingSystem/NanoDet')

from NanoDet.inference_img import *

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# 在脚本开始时初始化和加载模型
load_config(cfg, 'NanoDet/config/nanodet-plus-m-1.5x_416.yml')
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, 'NanoDet/nanodet-plus-m-1.5x_416_checkpoint.ckpt', logger, device="cpu")



class ImageProcessor(QMainWindow):
    def __init__(self):
        super(ImageProcessor, self).__init__()

        # 成员：
        self.original_image = None
        self.original_image_backup = None
        self.processed_image = None
        self.original_image_path = None
        self.processed_image_path = None
        self.choosed_algorithm = ""
        self.paramaters = {"直方图均衡化": None,  # 无参数
                           "频率域滤波": None,  # 低通、高通
                           "空域滤波": None,  # 均值、中值、高斯
                           "图像去噪": None,  # 中值、高斯
                           "小波变换": None,  # 无参数
                           "图像分割": None}

        # # 处理算法：
        # self.algorithms = ["直方图均衡化", "频率域滤波-低通", "频率域滤波-高通", "空域滤波-均值滤波",
        #                    "空域滤波-中值滤波", "空域滤波-高斯滤波", "图像去噪-中值滤波", "图像去噪-高斯滤波",
        #                    "小波变换", "图像分割-Canny", "图像分割-分水岭算法", "图像分割-GrabCut", "镜像-水平镜像"]
        self.algorithms = ["直方图均衡化", "频率域滤波", "空域滤波", "图像去噪", "小波变换", "图像分割","基于深度学习的目标检测"]




        self.init_ui()

        # 添加噪声对话框
        self.dialog_add = AddNoiseWidget(img=self.original_image_backup)
        # print("create dialog_add")
        # 将对话框关闭信号与主窗口的槽函数关联
        self.dialog_add.finished.connect(self.receive_noise_image)

    def init_ui(self):
        # 主布局和窗口设置
        self.setWindowTitle("图像处理系统")
        self.setGeometry(200, 200, 2000, 900)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        # 设置大小不变
        self.setFixedSize(self.width(), self.height())

        # 水平布局：图片展示区和控制面板
        hbox_layout = QHBoxLayout(central_widget)
        hbox_layout.setContentsMargins(40, 60, 40, 60)

        # 图片展示区布局（垂直布局）
        image_display_layout = QVBoxLayout()

        # 图片展示子区域（水平布局）：左侧原图，右侧处理后的图像
        image_display_sub_layout = QHBoxLayout()

        self.original_image_label = QLabel("原图")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("QLabel{border: 3px solid grey}")

        self.processed_image_label = QLabel("处理后的图像")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setStyleSheet("QLabel{border: 3px solid grey}")

        # 将图片标签添加到图片展示子区域的水平布局中
        image_display_sub_layout.addWidget(self.original_image_label)
        image_display_sub_layout.addWidget(self.processed_image_label)

        # 将图片展示子区域的水平布局添加到图片展示区的垂直布局中
        image_display_layout.addLayout(image_display_sub_layout)
        image_display_layout.setStretch(0, 4)  # 设置原图的宽度比例为 2
        image_display_layout.addSpacing(20)  # Add space between image display and parameter selection

        #### 参数选择区 ###

        self.parameter_selection_layout = QHBoxLayout()
        self.parameter_selection_layout.setContentsMargins(0, 0, 0, 0)

        # 创建combobox
        self.algo_choose_combo = QComboBox()
        self.algo_choose_combo.addItems(["低通", "高通"])
        self.parameter_selection_layout.addWidget(self.algo_choose_combo)
        # 设置高度，宽度
        self.algo_choose_combo.setMinimumHeight(self.height() // 16)
        self.algo_choose_combo.setMaximumWidth(self.width() // 6)
        # 设置所有的item居中
        for i in range(self.algo_choose_combo.count()):
            self.algo_choose_combo.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
        # 设置显示的item居中
        self.algo_choose_combo.setEditable(True)
        self.algo_choose_combo.lineEdit().setAlignment(Qt.AlignCenter)
        # 设置字体大小
        font = QFont()
        font.setPointSize(15)
        font.setFamily("楷体")
        self.algo_choose_combo.setFont(font)

        # 创建label
        self.parameter_label = QLabel("输入参数：")
        self.parameter_label.setAlignment(Qt.AlignCenter)
        self.parameter_label.setFont(font)
        self.parameter_selection_layout.addWidget(self.parameter_label)
        # 设置高度，宽度
        self.parameter_label.setMinimumHeight(self.height() // 16)
        self.parameter_label.setMaximumWidth(self.width() // 6)

        # 创建输入框
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("在此输入")
        self.parameter_selection_layout.addWidget(self.threshold_input)
        # 设置高度，宽度
        self.threshold_input.setMinimumHeight(self.height() // 16)
        self.threshold_input.setMaximumWidth(self.width() // 6)
        # 设置字体
        self.threshold_input.setFont(font)

        # 创建确认按钮
        self.confirm_button = QPushButton("确认")
        self.confirm_button.setFont(font)
        self.confirm_button.clicked.connect(self.process_image)
        self.confirm_button.setMinimumHeight(self.height() // 16)  # 设置最小高度
        self.confirm_button.setMaximumWidth(self.width() // 8)
        self.parameter_selection_layout.addWidget(self.confirm_button)

        # 所有控件均居左
        self.parameter_selection_layout.setAlignment(Qt.AlignLeft)


        # 禁用参数选择区
        self.prohibit_parameter_selection()

        ###########################################################################

        image_display_layout.addLayout(self.parameter_selection_layout)
        image_display_layout.setStretch(1, 1)  # 设置参数选择区的高度比例为 1

        # 将图片展示布局添加到主布局
        hbox_layout.addLayout(image_display_layout)
        hbox_layout.setStretch(0, 4)  # 设置图片展示区的宽度比例为 2

        # 控制面板
        control_panel = QVBoxLayout()
        # hbox_layout.addLayout(control_panel)
        # hbox_layout.setStretch(2, 1)  # 设置控制面板的宽度比例为 1
        # 将控制面板布局添加到主布局
        hbox_layout.addLayout(control_panel)
        hbox_layout.setStretch(1, 1)  # 设置控制面板的宽度比例为 1

        # 设置字体大小：15，微软雅黑
        font = QFont()
        font.setPointSize(15)
        font.setFamily("楷体")

        divide_number=14

        self.open_button = QPushButton("打开文件")
        self.open_button.setFont(font)
        self.open_button.clicked.connect(self.open_file)
        self.open_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.open_button)
        control_panel.addStretch(1)  # 添加弹性空间

        self.save_button = QPushButton("保存图像")
        self.save_button.setFont(font)
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.save_button)
        control_panel.addStretch(1)

        self.rotate_button = QPushButton("旋转图像")
        self.rotate_button.setFont(font)
        self.rotate_button.clicked.connect(self.rotate_image)
        self.rotate_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.rotate_button)
        control_panel.addStretch(1)

        self.mirror_button = QPushButton("镜像图像")
        self.mirror_button.setFont(font)
        self.mirror_button.clicked.connect(self.mirror_image)
        self.mirror_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.mirror_button)
        control_panel.addStretch(1)



        self.add_noise_button = QPushButton("添加噪声")
        self.add_noise_button.setFont(font)
        self.add_noise_button.clicked.connect(self.add_noise)
        self.add_noise_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.add_noise_button)
        control_panel.addStretch(1)


        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setFont(font)
        self.algorithm_combo.addItems(self.algorithms)
        self.algorithm_combo.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        # 设置所有的item居中
        for i in range(self.algorithm_combo.count()):
            self.algorithm_combo.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
        # 设置显示的item居中
        self.algorithm_combo.setEditable(True)
        self.algorithm_combo.lineEdit().setAlignment(Qt.AlignCenter)
        # 关联信号
        self.algorithm_combo.currentIndexChanged.connect(self.combo_changed)
        control_panel.addWidget(self.algorithm_combo)
        control_panel.addStretch(1)

        self.start_button = QPushButton("开始处理")
        self.start_button.setFont(font)
        self.start_button.clicked.connect(self.process_image)
        self.start_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.start_button)

        # 重置图像
        self.reset_button = QPushButton("重置图像")
        self.reset_button.setFont(font)
        self.reset_button.clicked.connect(self.recover_image)
        self.reset_button.setMinimumHeight(self.height() // divide_number)
        control_panel.addWidget(self.reset_button)

        self.clear_button = QPushButton("清除图像")
        self.clear_button.setFont(font)
        self.clear_button.clicked.connect(self.clear_image)
        self.clear_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.clear_button)
        control_panel.addStretch(1)

        self.visualize_button = QPushButton("可视化数据")
        self.visualize_button.setFont(font)
        self.visualize_button.clicked.connect(self.visualize_data)
        self.visualize_button.setMinimumHeight(self.height() // divide_number)  # 设置最小高度
        control_panel.addWidget(self.visualize_button)

        # 设置主窗口的布局
        central_widget.setLayout(hbox_layout)

        self.set_label_without_image()

    def prohibit_parameter_selection(self):
        # 将参数选择区清空，并且禁用
        # 把combo box的item清空，改成None
        self.algo_choose_combo.clear()
        self.algo_choose_combo.addItems(["None"])
        # 禁用
        self.algo_choose_combo.setEnabled(False)
        # 把label的text清空
        self.parameter_label.setText("暂无参数输入")
        # 把输入框的text清空
        self.threshold_input.setText("")
        # 禁用
        self.threshold_input.setEnabled(False)
        # 禁用
        self.confirm_button.setEnabled(False)

    def open_file(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "图片文件 (*.jpg *.png)")
        # 检查文件路径是否为空、是否为所需要的文件类型
        if file_path:
            # 首先清空图片
            self.clear_image()
            self.original_image_path = file_path
            self.original_image_backup = QPixmap(self.original_image_path)
            self.original_image = QPixmap(self.original_image_path)
            self.display_image(self.original_image_path)
        else:
            # # 警视窗口提示：未选择文件
            # QMessageBox.warning(self, "警告", "未选择文件", QMessageBox.Yes)
            return


    def save_image(self):
        # 保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "图片文件 (*.jpg *.png)")
        if file_path:
            self.processed_image_path = file_path
            self.processed_image.save(self.processed_image_path)
            # 弹窗提示：保存成功
            self.statusBar().showMessage("保存成功")
        else:
            # 弹窗提示：未选择文件
            self.statusBar().showMessage("未选择文件")

    def recover_image(self):
        # 恢复原图
        self.set_label_without_image()
        self.display_image(self.original_image_backup)
        self.original_image = self.original_image_backup
        self.processed_image = self.original_image_backup

    def add_noise(self):

        # print("here!!!")
        self.dialog_add.origin_image = self.original_image_backup
        self.dialog_add.image_processed = self.original_image_backup
        self.dialog_add.display_image(self.dialog_add.origin_image)
        self.dialog_add.show()


    def receive_noise_image(self):
        print("receive_noise_image")
        # 接收添加噪声后的图像
        self.original_image = self.dialog_add.added_noise_image
        # 如果不是Qimage对象，转换成Qimage对象
        if not isinstance(self.original_image, QImage):
            self.original_image = cv2qt(self.original_image)
        # print("type of processed_image: ", type(self.processed_image))
        # print("type of original_image: ", type(self.original_image))
        self.display_image(self.original_image)


    def combo_changed(self):
        # 获取当前选择的算法
        algorithm = self.algorithm_combo.currentText()

        # 根据算法选择添加参数选择
        if algorithm == "直方图均衡化":
            # 把combo box的item清空，改成“直方图均衡化”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["直方图均衡化"])
            # 禁用
            self.algo_choose_combo.setEnabled(False)
            # 把label的text改成“无参数输入”
            self.parameter_label.setText("无参数输入")
            # 清空输入框
            self.threshold_input.setText("")
            # 禁用
            self.threshold_input.setEnabled(False)
            # 启用
            self.confirm_button.setEnabled(True)



        elif algorithm == "频率域滤波":
            # 把combo box的item清空，改成“低通滤波”和“高通滤波”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["低通滤波", "高通滤波"])
            # 启用
            self.algo_choose_combo.setEnabled(True)
            # 把label的text改成“输入阈值：”
            self.parameter_label.setText("输入阈值：")
            # 设置默认值
            self.threshold_input.setText("20")
            # 启用
            self.threshold_input.setEnabled(True)
            # 启用
            self.confirm_button.setEnabled(True)

        elif algorithm == "空域滤波":
            # 把combo box的item清空，改成“均值滤波”和“中值滤波”和“高斯滤波”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["均值滤波", "中值滤波", "高斯滤波"])
            # 启用
            self.algo_choose_combo.setEnabled(True)
            # 把label的text改成“输入阈值：”
            self.parameter_label.setText("输入Kernel大小：")
            # 设置默认值
            self.threshold_input.setText("5")
            # 启用
            self.threshold_input.setEnabled(True)
            # 启用
            self.confirm_button.setEnabled(True)
        elif algorithm == "图像去噪":
            # 把combo box的item清空，改成“中值滤波”和“高斯滤波”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["中值滤波", "均值滤波", "高斯滤波"])
            # 启用
            self.algo_choose_combo.setEnabled(True)
            # 把label的text改成“输入阈值：”
            self.parameter_label.setText("输入Kernel大小：")
            # 设置默认值
            self.threshold_input.setText("5")
            # 启用
            self.threshold_input.setEnabled(True)
            # 启用
            self.confirm_button.setEnabled(True)
        elif algorithm == "小波变换":
            # 把combo box的item清空，改成“小波变换”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["小波变换"])
            # 禁用
            self.algo_choose_combo.setEnabled(False)
            # 把label的text改成“无参数输入”
            self.parameter_label.setText("无参数输入")
            # 清空输入框
            self.threshold_input.setText("")
            # 禁用
            self.threshold_input.setEnabled(False)
            # 启用
            self.confirm_button.setEnabled(True)


        elif algorithm == "图像分割":
            # 把combo box的item清空，改成“Canny”和“分水岭算法”和“GrabCut”
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["Canny", "分水岭算法"])
            # 启用
            self.algo_choose_combo.setEnabled(True)
            # 将label的text改成“无参数输入”
            self.parameter_label.setText("无参数输入")
            # 清空输入框
            self.threshold_input.setText("")
            # 禁用
            self.threshold_input.setEnabled(False)
            # 启用
            self.confirm_button.setEnabled(True)

        elif algorithm == "基于深度学习的目标检测":
            # 把combo box的item清空，改成"NanoDet"
            self.algo_choose_combo.clear()
            self.algo_choose_combo.addItems(["NanoDet"])
            # 禁用
            self.algo_choose_combo.setEnabled(False)
            # 把label的text改成“无参数输入”
            self.parameter_label.setText("无参数输入")
            # 清空输入框
            self.threshold_input.setText("")
            # 禁用
            self.threshold_input.setEnabled(False)
            # 启用
            self.confirm_button.setEnabled(True)



        else:
            pass

        self.update()

    def process_image(self):
        # 图像处理逻辑
        if not self.original_image:
            # 使用弹窗提示：未选择文件，而不是status bar
            # self.statusBar().showMessage("未选择文件")
            QMessageBox.warning(self, "警告", "未选择文件", QMessageBox.Yes)
            return
        algorithm = self.algorithm_combo.currentText()
        if algorithm == "直方图均衡化":
            self.processed_image = histogram_equalization_qimage(self.original_image)
            self.display_image(self.processed_image, is_original=False)
        elif algorithm == "频率域滤波":
            # 获取阈值
            threshold = self.threshold_input.text()
            # 转换为int
            threshold = int(threshold)
            # 获取选择的算法
            algorithm = self.algo_choose_combo.currentText()
            if algorithm == "低通滤波":
                self.processed_image = frequency_domain_filter_qimage(self.original_image, "lowpass",
                                                                      cutoff_frequency=threshold)
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "高通滤波":
                self.processed_image = frequency_domain_filter_qimage(self.original_image, "highpass",
                                                                      cutoff_frequency=threshold)
                self.display_image(self.processed_image, is_original=False)
        elif algorithm == "空域滤波":
            # 获取阈值
            threshold = self.threshold_input.text()
            # 转换为int
            threshold = int(threshold)
            # 获取选择的算法
            algorithm = self.algo_choose_combo.currentText()
            if algorithm == "均值滤波":
                self.processed_image = spatial_domain_filtering_qimage(self.original_image, "mean",
                                                                       kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "中值滤波":
                self.processed_image = spatial_domain_filtering_qimage(self.original_image, "median",
                                                                       kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "高斯滤波":
                self.processed_image = spatial_domain_filtering_qimage(self.original_image, "gaussian",
                                                                       kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
        elif algorithm == "图像去噪":
            # 获取阈值
            threshold = self.threshold_input.text()
            # 转换为int
            threshold = int(threshold)
            # 获取选择的算法
            algorithm = self.algo_choose_combo.currentText()
            if algorithm == "中值滤波":
                self.processed_image = denoising_qimage(self.original_image, "median", kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "均值滤波":
                self.processed_image = denoising_qimage(self.original_image, "mean", kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "高斯滤波":
                self.processed_image = denoising_qimage(self.original_image, "gaussian", kernel_size=threshold)
                self.display_image(self.processed_image, is_original=False)
        elif algorithm == "小波变换":
            self.processed_image = wavelet_transform_qimage(self.original_image)
            self.display_image(self.processed_image, is_original=False)
        elif algorithm == "图像分割":
            # 获取选择的算法
            algorithm = self.algo_choose_combo.currentText()
            if algorithm == "Canny":
                self.processed_image = image_segmentation_qimage(self.original_image, "canny")
                self.display_image(self.processed_image, is_original=False)
            elif algorithm == "分水岭算法":
                self.processed_image = image_segmentation_qimage(self.original_image, "watershed")
                self.display_image(self.processed_image, is_original=False)

        elif algorithm == "基于深度学习的目标检测":
            # 首先将图片暂时保存到本地
            self.original_image.save("temp.jpg")
            # 进行目标检测
            result_image=detect_image("temp.jpg",predictor)
            # 转换成QImage对象
            self.processed_image=cv2qt(result_image)
            # 显示图像
            self.display_image(self.processed_image, is_original=False)
            # 删除临时文件
            os.remove("temp.jpg")


        else:
            pass

    def set_label_without_image(self):
        self.original_image_label.setText("原图")
        self.processed_image_label.setText("处理后")
        # 设置标签文本居中
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setAlignment(Qt.AlignCenter)

        # 设置字体大小：20,字体颜色灰色
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        font.setFamily("微软雅黑")
        # 设置标签的边框
        self.original_image_label.setStyleSheet("QLabel{border: 3px solid grey; color: grey;background-color: white}")
        self.processed_image_label.setStyleSheet("QLabel{border: 3px solid grey; color: grey;background-color: white}")
        self.original_image_label.setFont(font)
        self.processed_image_label.setFont(font)
        self.update()

    # 把两个图的大小设置成一样的
    def set_image_same_size(self):
        if self.original_image and self.processed_image:
            # 获取两个图的大小
            original_size = self.original_image.size()
            processed_size = self.processed_image.size()
            # 获取两个图的宽度和高度
            original_width = original_size.width()
            original_height = original_size.height()
            processed_width = processed_size.width()
            processed_height = processed_size.height()
            # 获取高度最小值
            min_height = min(original_height, processed_height)
            # 获取宽度最小值
            min_width = min(original_width, processed_width)
            # 将图片的大小设置成一样的，而不是标签
            self.original_image = self.original_image.scaled(min_width, min_height, Qt.KeepAspectRatio,
                                                             Qt.SmoothTransformation)
            self.processed_image = self.processed_image.scaled(min_width, min_height, Qt.KeepAspectRatio,
                                                               Qt.SmoothTransformation)

    def display_image(self, img, is_original=True):
        # 判断 img 是图像路径还是 QImage 对象
        if isinstance(img, str):
            pixmap = QPixmap(img)  # 从文件路径加载
        elif isinstance(img, QImage):
            pixmap = QPixmap.fromImage(img)  # 从 QImage 对象加载
        elif isinstance(img, QPixmap):
            pixmap = img
        else:
            print("Invalid image format")
            return

        if is_original:
            self.original_image = pixmap
        else:
            self.processed_image = pixmap

        # 把两个图的大小设置成一样的
        self.set_image_same_size()

        # 调整 pixmap 的大小以适应标签，同时保持长宽比
        # 假设 self.original_image_label 和 self.processed_image_label 是两个用于显示图像的 QLabel
        if is_original:
            label = self.original_image_label
        else:
            label = self.processed_image_label

        # 获取标签的大小，并保持图片长宽比
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 将调整后的 pixmap 设置为标签的 pixmap
        label.setPixmap(scaled_pixmap)
        # 把标签大小设置为调整后的 pixmap 的大小
        label.resize(scaled_pixmap.size())

    def rotate_image(self):

        angle = 90  # 旋转角度

        # 旋转原图
        if self.original_image_label.pixmap() and not self.original_image_label.pixmap().isNull():
            self.rotate_and_update_label(self.original_image_label, self.original_image, angle)

        # 旋转处理后的图像
        if self.processed_image_label.pixmap() and not self.processed_image_label.pixmap().isNull():
            self.rotate_and_update_label(self.processed_image_label, self.processed_image, angle)

        self.update()

    def mirror_image(self):
        # 镜像原图
        if self.original_image_label.pixmap() and not self.original_image_label.pixmap().isNull():
            self.mirror_and_update_label(self.original_image_label, self.original_image)

        # 镜像处理后的图像
        if self.processed_image_label.pixmap() and not self.processed_image_label.pixmap().isNull():
            self.mirror_and_update_label(self.processed_image_label, self.processed_image)

        self.update()

    def rotate_and_update_label(self, label, pixmap, angle):
        # 创建一个 QTransform 对象并应用旋转
        transform = QTransform()
        transform.rotate(angle)

        # 使用 transform 对 pixmap 进行旋转
        rotated_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

        # 更新标签中显示的图片
        label.setPixmap(rotated_pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新图片
        if label == self.original_image_label:
            self.original_image = rotated_pixmap
        else:
            self.processed_image = rotated_pixmap

    def mirror_and_update_label(self, label, pixmap):
        # 创建一个 QTransform 对象并应用旋转
        transform = QTransform()
        transform.scale(-1, 1)

        # 使用 transform 对 pixmap 进行旋转
        mirrored_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

        # 更新标签中显示的图片
        label.setPixmap(mirrored_pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新图片
        if label == self.original_image_label:
            self.original_image = mirrored_pixmap
        else:
            self.processed_image = mirrored_pixmap

    def clear_image(self):
        self.original_image_label.clear()
        self.processed_image_label.clear()
        self.original_image = None
        self.processed_image = None
        self.original_image_backup = None
        self.original_image_path = None
        self.processed_image_path = None
        self.set_label_without_image()
        self.update()

    # def add_noise(self):
    # 添加噪声

    def visualize_data(self):
        # 可视化数据
        if self.original_image and self.processed_image:
            self.dialog_visualize = VisualizeDataWidget(self.original_image, self.processed_image)
            print("create dialog_visualize")
            self.dialog_visualize.display_data()
            print("display image finished")
            self.dialog_visualize.show()
        else:
            # 使用弹窗提示：未选择文件，而不是status bar
            # self.statusBar().showMessage("未选择文件")
            QMessageBox.warning(self, "警告", "未选择图片或未处理完成！", QMessageBox.Yes)
            return




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = ImageProcessor()
    main_win.show()
    sys.exit(app.exec_())
