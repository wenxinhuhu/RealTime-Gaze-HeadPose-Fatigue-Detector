import threading
import datetime
from playsound import playsound
import platform
system = platform.system()
if system != "Windows":
    import vlc
import time
import cv2  # 图像处理库 OpenCV
import dlib  # 人脸检测库 dlib
import imutils
import numpy as np  # 数据处理库 NumPy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QMessageBox, QDesktopWidget
from imutils import face_utils
from scipy.spatial import distance as dist

# 点头/摇头检测参数和光流法跟踪配置
lk_params = dict(winSize=(35, 35), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
NOD_THRESHOLD = 20        # 点头垂直位移阈值
SHAKE_THRESHOLD = 10      # 摇头水平位移阈值
DIRECTION_CHANGES = 2     # 摇头方向变换次数阈值
DOMINANCE_RATIO = 1.5     # 主导方向比例阈值
TRACKING_FRAMES = 15      # 光流轨迹帧数窗口
MIN_MOVE = 2              # 判定有效移动的最小像素
DISPLAY_DURATION = 45     # 检测到动作后文本显示持续帧数

def get_coords(p):
    """提取光流跟踪点的坐标"""
    try:
        return int(p[0][0][0]), int(p[0][0][1])
    except:
        return int(p[0][0]), int(p[0][1])

class GestureDetector:
    """手势（点头/摇头）检测器，维护光流跟踪点和运动分析"""
    def __init__(self):
        self.track_points = []               # 光流跟踪点历史
        self.direction_history = []          # 水平运动方向历史（用于判断摇头）
        self.gesture_status = {"nod": 0, "shake": 0}  # 动作状态计数，用于控制显示
        self.face_center = None              # 当前跟踪的人脸中心点
        self.lost_counter = 0               # 连续跟踪丢失计数
        self.debug_info = {}                # 调试信息

    def update_tracking(self, new_point):
        """更新光流跟踪点位置历史"""
        self.track_points.append(new_point)
        if len(self.track_points) > TRACKING_FRAMES:
            self.track_points.pop(0)
        if len(self.track_points) >= 2:
            prev = self.track_points[-2]
            curr = self.track_points[-1]
            dx = curr[0][0][0] - prev[0][0][0]
            # 如果水平移动超过最小阈值，则记录方向（1=右移，-1=左移）
            if abs(dx) > MIN_MOVE:
                direction = 1 if dx > 0 else -1
                self.direction_history.append(direction)
                if len(self.direction_history) > 20:
                    self.direction_history = self.direction_history[-20:]

    def analyze_motion(self):
        """分析累计的轨迹位移，判断是否构成点头或摇头动作"""
        if len(self.track_points) < 2:
            return None
        start_point = self.track_points[0]
        end_point = self.track_points[-1]
        total_x = end_point[0][0][0] - start_point[0][0][0]
        total_y = end_point[0][0][1] - start_point[0][0][1]
        # 保存调试信息
        self.debug_info = {
            'total_x': total_x,
            'total_y': total_y,
            'abs_x': abs(total_x),
            'abs_y': abs(total_y)
        }
        # 位移不足最小运动量，不判定为任何动作
        min_motion = max(NOD_THRESHOLD, SHAKE_THRESHOLD) * 0.6
        if abs(total_x) + abs(total_y) < min_motion:
            return None
        # 计算水平/垂直位移占比
        sum_xy = abs(total_x) + abs(total_y)
        x_ratio = abs(total_x) / sum_xy if sum_xy != 0 else 0
        y_ratio = abs(total_y) / sum_xy if sum_xy != 0 else 0
        is_x_dominant = x_ratio > (y_ratio * DOMINANCE_RATIO)
        is_y_dominant = y_ratio > (x_ratio * DOMINANCE_RATIO)
        # 判断点头：垂直方向位移占主导且超过阈值
        if is_y_dominant and abs(total_y) > NOD_THRESHOLD:
            return "nod"
        # 判断摇头：水平方向位移占主导且超过阈值，且累计方向变换次数达到要求
        if is_x_dominant and abs(total_x) > SHAKE_THRESHOLD:
            dir_changes = sum(1 for i in range(1, len(self.direction_history))
                              if self.direction_history[i] != self.direction_history[i-1])
            if dir_changes >= DIRECTION_CHANGES:
                return "shake"
        return None

    def reset_counters(self, detected_gesture):
        """检测到动作后重置计数器，触发显示计数"""
        if detected_gesture == "nod":
            self.gesture_status["nod"] = DISPLAY_DURATION
            self.direction_history = []  # 点头时清空方向变化历史
        elif detected_gesture == "shake":
            self.gesture_status["shake"] = DISPLAY_DURATION
            # 摇头后保留最近几个轨迹点，避免长期轨迹导致重复判断
            self.track_points = self.track_points[-5:]

class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.qmessagebox = QMessageBox()
        MainWindow.setObjectName("MainWindow")
        # 居中窗口并设置初始大小
        self.desktop = QApplication.desktop()
        MainWindow.resize(1000, 600)
        screen = QDesktopWidget().screenGeometry()
        size = MainWindow.geometry()
        MainWindow.move((screen.width() - size.width()) // 2,
                        (screen.height() - size.height()) // 2)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.Box_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.Box_5.setEnabled(True)
        # 主容器样式（背景色）
        self.Box_5.setObjectName("Box_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.Box_5)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.Box_5)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_5.addWidget(self.pushButton_4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_5.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.Box_5)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        # 疲劳检测参数标签和输入框
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(101, 16))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox.setStyleSheet("font: 9pt 'Microsoft YaHei';")
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_3.addWidget(self.spinBox)
        self.horizontalLayout3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout3.setObjectName("horizontalLayout3")
        self.label2 = QtWidgets.QLabel(self.groupBox_2)
        self.label2.setMaximumSize(QtCore.QSize(101, 16))
        self.label2.setObjectName("label2")
        self.horizontalLayout3.addWidget(self.label2)
        self.spinBox1 = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox1.setStyleSheet("font: 9pt 'Microsoft YaHei';")
        self.spinBox1.setObjectName("spinBox1")
        self.horizontalLayout3.addWidget(self.spinBox1)
        self.horizontalLayout4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout4.setObjectName("horizontalLayout4")
        self.label3 = QtWidgets.QLabel(self.groupBox_2)
        self.label3.setMaximumSize(QtCore.QSize(101, 16))
        self.label3.setObjectName("label3")
        self.horizontalLayout4.addWidget(self.label3)
        self.spinBox2 = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox2.setStyleSheet("font: 9pt 'Microsoft YaHei';")
        self.spinBox2.setObjectName("spinBox2")
        self.horizontalLayout4.addWidget(self.spinBox2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout3)
        self.verticalLayout_2.addLayout(self.horizontalLayout4)
        # 疲劳检测选项复选框（打哈欠、监测数据、闭眼）
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_2.addWidget(self.checkBox)
        self.checkBox3 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox3.setObjectName("checkBox3")
        self.horizontalLayout_2.addWidget(self.checkBox3)
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_2.setObjectName("checkBox_2")
        self.horizontalLayout_2.addWidget(self.checkBox_2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        # 新增：点头/摇头检测选项复选框
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.checkBox_nod = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_nod.setObjectName("checkBox_nod")
        self.horizontalLayout_9.addWidget(self.checkBox_nod)
        self.checkBox_shake = QtWidgets.QCheckBox(self.groupBox_2)
        self.checkBox_shake.setObjectName("checkBox_shake")
        self.horizontalLayout_9.addWidget(self.checkBox_shake)
        self.verticalLayout_2.addLayout(self.horizontalLayout_9)
        self.verticalLayout_5.addWidget(self.groupBox_2)
        # 脱离范围检测分组
        self.groupBox_5 = QtWidgets.QGroupBox(self.Box_5)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.checkBox_4 = QtWidgets.QCheckBox(self.groupBox_5)
        self.checkBox_4.setObjectName("checkBox_4")
        self.horizontalLayout_7.addWidget(self.checkBox_4)
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setMaximumSize(QtCore.QSize(101, 16))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_7.addWidget(self.label_3)
        self.spinBox_2 = QtWidgets.QSpinBox(self.groupBox_5)
        self.spinBox_2.setStyleSheet("font: 9pt 'Microsoft YaHei';")
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout_7.addWidget(self.spinBox_2)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_5.addWidget(self.groupBox_5)
        # 状态输出分组（文本日志）
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.groupBox_4 = QtWidgets.QGroupBox(self.Box_5)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_4)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout_5.addWidget(self.groupBox_4)
        self.horizontalLayout_6.addLayout(self.verticalLayout_5)
        # 视频显示区域
        self.graphicsView = QtWidgets.QLabel(self.Box_5)
        self.graphicsView.setMaximumSize(QtCore.QSize(self.desktop.width(), self.desktop.height()))
        self.graphicsView.setContentsMargins(2, 2, 2, 2)
        self.graphicsView.setScaledContents(True)
        self.width = self.graphicsView.width()
        self.height = self.graphicsView.height()
        self.graphicsView.setObjectName("graphicsView")
        # 加载默认封面图像
        self.COVER = 'images/initiate.png'
        self.img1 = cv2.imread(self.COVER)
        image1 = cv2.pyrDown(self.img1)
        y, x = image1.shape[:-1]
        self.cvimg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        self.frame1 = QImage(self.cvimg1, x, y, QImage.Format_RGB888)
        self.pix = QPixmap.fromImage(self.frame1)
        self.scene = QGraphicsScene()
        self.graphicsView.setPixmap(self.pix)
        self.graphicsView.show()
        self.scene.addPixmap(self.pix)
        self.horizontalLayout_6.addWidget(self.graphicsView)
        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 7)
        self.verticalLayout_6.addWidget(self.Box_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.retranslateUi(MainWindow)
        self.connect()
        # 设置统一的界面风格与现代配色
        self.centralwidget.setStyleSheet("font: 10pt 'Microsoft YaHei'; background-color: #F0F0F0;")
        self.groupBox.setStyleSheet("QGroupBox { font: bold 10pt 'Microsoft YaHei'; border: 1px solid #CCCCCC; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 5px; }")
        self.groupBox_2.setStyleSheet("QGroupBox { font: bold 10pt 'Microsoft YaHei'; border: 1px solid #CCCCCC; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 5px; }")
        self.groupBox_5.setStyleSheet("QGroupBox { font: bold 10pt 'Microsoft YaHei'; border: 1px solid #CCCCCC; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 5px; }")
        self.groupBox_4.setStyleSheet("QGroupBox { font: bold 10pt 'Microsoft YaHei'; border: 1px solid #CCCCCC; border-radius: 5px; margin-top: 10px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 5px; }")
        self.pushButton_2.setStyleSheet("background-color: #4CAF50; color: white; font: 9pt 'Microsoft YaHei'; padding: 5px 10px; border: none; border-radius: 5px;")
        self.pushButton_4.setStyleSheet("background-color: #FFA500; color: white; font: 9pt 'Microsoft YaHei'; padding: 5px 10px; border: none; border-radius: 5px;")
        self.textBrowser.setStyleSheet("background-color: #FFFFFF; font: 9pt 'Microsoft YaHei';")
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "视觉显示系统"))
        self.Box_5.setTitle(_translate("MainWindow", "播放"))
        self.groupBox.setTitle(_translate("MainWindow", "视频源"))
        self.pushButton_2.setText(_translate("MainWindow", "开始检测"))
        self.pushButton_4.setText(_translate("MainWindow", "暂停"))
        self.groupBox_2.setTitle(_translate("MainWindow", "疲劳检测"))
        self.label_2.setText(_translate("MainWindow", "闭眼时间(帧):"))
        self.label3.setText(_translate("MainWindow", "间隔时间(秒):"))
        self.label2.setText(_translate("MainWindow", "哈欠次数:"))
        self.checkBox.setText(_translate("MainWindow", "打哈欠"))
        self.checkBox_2.setText(_translate("MainWindow", "闭眼"))
        self.checkBox3.setText(_translate("MainWindow", "监测数据"))
        self.checkBox_nod.setText(_translate("MainWindow", "点头"))
        self.checkBox_shake.setText(_translate("MainWindow", "摇头"))
        self.groupBox_5.setTitle(_translate("MainWindow", "脱离范围检测"))
        self.checkBox_4.setText(_translate("MainWindow", "脱离范围检测"))
        self.label_3.setText(_translate("MainWindow", "脱离时间(帧):"))
        self.groupBox_4.setTitle(_translate("MainWindow", "状态输出"))
        # 默认选中各检测功能
        self.checkBox.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox3.setChecked(True)
        self.checkBox_nod.setChecked(True)
        self.checkBox_shake.setChecked(True)
        self.checkBox_4.setChecked(True)

    def connect(self):
        self.mtt = []
        self.pushButton_2.clicked.connect(self.camera_on)
        self.pushButton_4.clicked.connect(self.off)
        # 硬编码使用本地摄像头
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False 表示摄像头未打开，True表示已打开
        # 闭眼判定连续帧数阈值
        self.AR_CONSEC_FRAMES_check = 60
        # 脱离范围判定连续帧数阈值
        self.OUT_AR_CONSEC_FRAMES_check = 90
        self.spinBox.valueChanged.connect(self.AR_CONSEC_FRAMES)
        self.spinBox_2.valueChanged.connect(self.OUT_AR_CONSEC_FRAMES)
        self.spinBox1.valueChanged.connect(self.Number_Of_Yawns_Judged_As_Fatigue_2)
        self.spinBox2.valueChanged.connect(self.NOYJAF_Time_2)
        # 眼睛长宽比阈值
        self.EYE_AR_THRESH = 0.24
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 打哈欠嘴部长宽比阈值
        self.MAR_THRESH = 0.6
        self.MOUTH_AR_CONSEC_FRAMES = 30
        self.time_reduce = 0
        # 初始化设置数值
        self.spinBox_2.setValue(self.OUT_AR_CONSEC_FRAMES_check)
        self.spinBox.setValue(self.AR_CONSEC_FRAMES_check)
        # 计数器初始化
        self.COUNTER = 0
        self.TOTAL = 0
        self.eToTAL = 0
        self.PERCLOS = 0.12
        self.TIME_PERCLOS = self.AR_CONSEC_FRAMES_check
        self.mCOUNTER = 0
        self.mTOTAL = 0
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.oCOUNTER = 0
        self.Number_Of_Yawns_Judged_As_Fatigue = 4
        self.NOYJAF_Time = 180
        self.spinBox1.setValue(self.Number_Of_Yawns_Judged_As_Fatigue)
        self.spinBox2.setValue(self.NOYJAF_Time)
        self.shutEye = False
        self.ifYawming = False
        self.ifTired = False
        self.ifNoFace = False
        # 将初始哈欠时间设置为过去时间，以便第一次计算间隔
        self.timeOfTheLastOfYawns = datetime.datetime(2022, 12, 31)
        self.timeOfTheFirstOfYawns = datetime.datetime(2022, 12, 31)

    def __del__(self):
        pass

    def AR_CONSEC_FRAMES(self, event):
        self.textBrowser.append(u"设置疲劳间隔为:\t" + str(self.spinBox.value()) + " 秒")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.AR_CONSEC_FRAMES_check = int(self.spinBox.value())
        # 更新闭眼持续帧数阈值
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check

    def OUT_AR_CONSEC_FRAMES(self, event):
        self.textBrowser.append(u"设置脱离识别间隔为:\t" + str(self.spinBox_2.value()) + " 秒")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.OUT_AR_CONSEC_FRAMES_check = int(self.spinBox_2.value())

    def Number_Of_Yawns_Judged_As_Fatigue_2(self, event):
        self.textBrowser.append(u"设置哈欠次数为:\t" + str(self.spinBox1.value()))
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.Number_Of_Yawns_Judged_As_Fatigue = int(self.spinBox1.value())

    def NOYJAF_Time_2(self, event):
        self.textBrowser.append(u"设置间隔时间为:\t" + str(self.spinBox2.value()))
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.NOYJAF_Time = int(self.spinBox2.value())

    def camera_on(self, event):
        import _thread
        if self.CAMERA_STYLE:
            return  # 如果摄像头已在运行，则不重复打开
        _thread.start_new_thread(self._learning_face, (event,))

    def off(self):
        # 停止摄像头采集
        if self.CAMERA_STYLE:
            self.CAMERA_STYLE = False
            if hasattr(self, 'cap'):
                self.cap.release()
            self.graphicsView.setPixmap(self.pix)
        # 重置检测状态
        self.COUNTER = self.mCOUNTER = self.hCOUNTER = self.oCOUNTER = 0
        self.TOTAL = self.eToTAL = self.mTOTAL = self.hTOTAL = 0
        self.shutEye = self.ifYawming = self.ifTired = self.ifNoFace = False
        self.timeOfTheLastOfYawns = datetime.datetime(2022, 12, 31)
        self.timeOfTheFirstOfYawns = datetime.datetime(2022, 12, 31)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])
        B = np.linalg.norm(mouth[4] - mouth[7])
        C = np.linalg.norm(mouth[0] - mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar

    def spredictor(self):
        # 加载人脸检测器和68点特征预测模型
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    def _learning_face(self, event):
        # 初始化人脸特征检测
        self.spredictor()
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        # 打开摄像头视频流
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return
        else:
            self.CAMERA_STYLE = True
        self.time_start = time.perf_counter()
        # 初始化手势检测器和光流前一帧
        self.gesture_detector = GestureDetector()
        prev_gray = None
        # 摄像头读取循环
        while self.cap.isOpened():
            if not self.CAMERA_STYLE:
                # 用户触发暂停，退出循环
                break
            flag, im_rd = self.cap.read()
            if not flag:
                continue
            # 调整帧尺寸，加快处理
            im_rd = imutils.resize(im_rd, 240)
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            if len(faces) != 0:
                # **光流跟踪点头/摇头检测**
                if self.checkBox_nod.isChecked() or self.checkBox_shake.isChecked():
                    # 取第一张人脸用于头部动作检测
                    d0 = faces[0]
                    x = d0.left(); y = d0.top()
                    w = d0.right() - d0.left(); h = d0.bottom() - d0.top()
                    # 如果未初始化跟踪点或跟踪中断超过5帧，则重新初始化跟踪点
                    if self.gesture_detector.face_center is None or self.gesture_detector.lost_counter > 5:
                        face_center = (x + w // 2, y + h // 2 + h // 5)
                        self.gesture_detector.face_center = face_center
                        self.gesture_detector.track_points = [np.array([[face_center]], np.float32)]
                        self.gesture_detector.lost_counter = 0
                        prev_gray = img_gray.copy()
                    else:
                        self.gesture_detector.lost_counter = 0
                    # 若存在跟踪点则计算光流
                    if self.gesture_detector.track_points:
                        try:
                            # 计算光流跟踪的新位置
                            new_points, st, err = cv2.calcOpticalFlowPyrLK(
                                prev_gray, img_gray, self.gesture_detector.track_points[-1], None, **lk_params)
                            if new_points is not None:
                                self.gesture_detector.update_tracking(new_points)
                                prev_gray = img_gray.copy()
                                # 分析运动轨迹判断动作
                                gesture = self.gesture_detector.analyze_motion()
                                # 根据选中选项过滤未启用的手势检测
                                if gesture == "nod" and not self.checkBox_nod.isChecked():
                                    gesture = None
                                if gesture == "shake" and not self.checkBox_shake.isChecked():
                                    gesture = None
                                if gesture:
                                    if (gesture == "nod" and self.checkBox_nod.isChecked()) or (gesture == "shake" and self.checkBox_shake.isChecked()):
                                        self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + ("点头" if gesture == "nod" else "摇头"))
                                        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                                        t = threading.Thread(target=self.nod if gesture == "nod" else self.shake)
                                        t.start()
                                        self.gesture_detector.reset_counters(gesture)
                        except Exception as e:
                            print(f"Tracking error: {str(e)}")
                            self.gesture_detector.track_points = []
                            self.gesture_detector.face_center = None
                # 在画面上显示检测结果（点头/摇头提示文字）
                if self.gesture_detector.gesture_status["nod"] > 0:
                    cv2.putText(im_rd, "NODDING", (50, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    self.gesture_detector.gesture_status["nod"] -= 1
                if self.gesture_detector.gesture_status["shake"] > 0:
                    cv2.putText(im_rd, "SHAKING", (50, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                    self.gesture_detector.gesture_status["shake"] -= 1
                # 如果启用了监测数据，则绘制跟踪点供调试观察
                if self.checkBox3.isChecked():
                    for point in self.gesture_detector.track_points:
                        px, py = get_coords(point)
                        cv2.circle(im_rd, (px, py), 3, (0, 255, 0), -1)
                # **疲劳状态检测（打哈欠/闭眼）**
                for k, d in enumerate(faces):
                    # 绘制人脸矩形框（仅调试模式下显示）
                    if self.checkBox3.isChecked():
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 提取人脸特征点
                    shape = self.predictor(im_rd, d)
                    if self.checkBox3.isChecked():
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    shape = face_utils.shape_to_np(shape)
                    # 下巴和鼻尖坐标差，用于辅助判断（ooo为调试用途）
                    ll = (shape[67, 0], shape[67, 1])
                    oo = (shape[63, 0], shape[63, 1])
                    self.ooo = -(ll[0] - oo[0]) + ll[1] - oo[1]
                    # 打哈欠检测
                    if self.checkBox.isChecked():
                        mouth = shape[mStart:mEnd]
                        mar = self.mouth_aspect_ratio(mouth)
                        mouthHull = cv2.convexHull(mouth)
                        if self.checkBox3.isChecked():
                            cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                        if mar > self.MAR_THRESH:
                            self.mCOUNTER += 1
                        if mar < self.MAR_THRESH:
                            self.mCOUNTER = 0
                        else:
                            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                                self.ifYawming = True
                        if self.ifYawming:
                            if self.mTOTAL == 0:
                                self.timeOfTheFirstOfYawns = datetime.datetime.now()
                            if self.mTOTAL == self.Number_Of_Yawns_Judged_As_Fatigue - 1:
                                self.timeOfTheLastOfYawns = datetime.datetime.now()
                            if mar < self.MAR_THRESH:
                                self.ifYawming = False
                                self.mTOTAL += 1
                                self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + "打哈欠")
                            if self.checkBox3.isChecked():
                                cv2.putText(im_rd, "Yawning", (100, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # （可选）显示打哈欠相关调试信息
                        if hasattr(self, 'putText'):
                            self.putText(im_rd, mar)
                        if self.time_reduce >= 60 and self.ooo < 20:
                            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                    # 闭眼（打瞌睡）检测
                    if self.checkBox_2.isChecked():
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        if self.checkBox3.isChecked():
                            cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                            cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        self.time_end = time.perf_counter()
                        self.time_reduce = self.time_end - self.time_start
                        if ear < self.EYE_AR_THRESH:
                            self.COUNTER += 1
                        if ear > self.EYE_AR_THRESH:
                            self.COUNTER = 0
                        else:
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                                self.shutEye = True
                        # 综合判断疲劳（根据哈欠次数和时间间隔）
                        if self.mTOTAL >= self.Number_Of_Yawns_Judged_As_Fatigue and (0 < (self.timeOfTheLastOfYawns - self.timeOfTheFirstOfYawns).seconds < self.NOYJAF_Time):
                            self.ifTired = True
                        if self.checkBox3.isChecked():
                            cv2.putText(im_rd, f"FACES: {len(faces)}", (5, 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            cv2.putText(im_rd, f"COUNTER: {self.COUNTER}", (70, 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (5, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # 未检测到人脸的情况
                if self.checkBox_nod.isChecked() or self.checkBox_shake.isChecked():
                    # 计数跟踪丢失帧数
                    self.gesture_detector.lost_counter += 1
                    self.gesture_detector.face_center = None
                    if self.gesture_detector.lost_counter > 5:
                        self.gesture_detector.track_points = []
                self.oCOUNTER += 1
                if self.checkBox3.isChecked():
                    cv2.putText(im_rd, "No Face", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + "脱离识别范围!!!")
                    self.ifNoFace = True
                    t = threading.Thread(target=self.noFace)
                    t.start()
                    self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                    self.oCOUNTER = 0
            # 疲劳事件报警及日志记录
            if self.shutEye:
                self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + "睡觉")
                t = threading.Thread(target=self.sleep)
                t.start()
                self.shutEye = False
                self.COUNTER = 0
            if self.ifTired:
                # 重置相关计数
                self.TOTAL = 0
                self.mTOTAL = 0
                self.hTOTAL = 0
                self.eToTAL = 0
                self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + "疲劳")
                t = threading.Thread(target=self.tired)
                t.start()
                self.ifTired = False
                self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
            # 将处理后的帧转换为QImage在界面上显示
            image1 = cv2.pyrUp(im_rd)
            y, x = image1.shape[:-1]
            self.width = self.centralwidget.width()
            self.height = self.centralwidget.height()
            # 根据窗口大小选择缩放策略
            if (self.width - self.width * 1 / 7) > x and False:  # 恒为False，仅供日后扩展
                cvimg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                y, x = image1.shape[:2]
                frame1 = QImage(cvimg1, x, y, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame1)
                self.graphicsView.setPixmap(pix)
                time.sleep(0.0015)
            else:
                cvimg1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
                y, x = im_rd.shape[:2]
                frame1 = QImage(cvimg1, x, y, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame1)
                self.graphicsView.setPixmap(pix)
        # 退出循环后，释放摄像头
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def noFace(self):
        if system == "Windows":
            playsound("NoFace.mp3")
        else:
            player = vlc.MediaPlayer("NoFace.mp3")
            player.play()

    def tired(self):
        if system == "Windows":
            playsound("Tired.mp3")
        else:
            player1 = vlc.MediaPlayer("Tired.mp3")
            player1.play()

    def sleep(self):
        if system == "Windows":
            playsound("Sleep.mp3")
        else:
            player2 = vlc.MediaPlayer("Sleep.mp3")
            player2.play()

    # 点头动作提示音
    def nod(self):
        if system == "Windows":
            playsound("Nod.mp3")
        else:
            player = vlc.MediaPlayer("Nod.mp3")
            player.play()

    # 摇头动作提示音
    def shake(self):
        if system == "Windows":
            playsound("Shake.mp3")
        else:
            player = vlc.MediaPlayer("Shake.mp3")
            player.play()

# 主程序入口
if __name__ == '__main__':
    app = QApplication([])
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    # 启动事件循环
    app.exec()
