import threading
import datetime
from playsound import playsound
import platform
system = platform.system()
if system != "Windows":
    import vlc
import time
import cv2  # 图像处理的库OpenCv
import dlib  # 人脸识别的库dlib
import imutils
import numpy as np  # 数据处理的库 numpy
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QGraphicsScene, QMessageBox, QDesktopWidget
from imutils import face_utils
from scipy.spatial import distance as dist

class Ui_MainWindow(object):
    def __init__(self, MainWindow):
        self.qmessagebox = QMessageBox()
        MainWindow.setObjectName("MainWindow")
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
        self.Box_5.setStyleSheet("Box_5{\n"
                                 "    rgb(199, 135, 255)\n"
                                 "\n"
                                 "}")
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
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(101, 16))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox.setStyleSheet("font: 9pt \"Arial\";")
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_3.addWidget(self.spinBox)
        self.horizontalLayout3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout3.setObjectName("horizontalLayout3")
        self.label2 = QtWidgets.QLabel(self.groupBox_2)
        self.label2.setMaximumSize(QtCore.QSize(101, 16))
        self.label2.setObjectName("label2")
        self.horizontalLayout3.addWidget(self.label2)
        self.spinBox1 = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox1.setStyleSheet("font: 9pt \"Arial\";")
        self.spinBox1.setObjectName("spinBox1")
        self.horizontalLayout3.addWidget(self.spinBox1)
        self.horizontalLayout4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout4.setObjectName("horizontalLayout4")
        self.label3 = QtWidgets.QLabel(self.groupBox_2)
        self.label3.setMaximumSize(QtCore.QSize(101, 16))
        self.label3.setObjectName("label3")
        self.horizontalLayout4.addWidget(self.label3)
        self.spinBox2 = QtWidgets.QSpinBox(self.groupBox_2)
        self.spinBox2.setStyleSheet("font: 9pt \"Arial\";")
        self.spinBox2.setObjectName("spinBox2")
        self.horizontalLayout4.addWidget(self.spinBox2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addLayout(self.horizontalLayout3)
        self.verticalLayout_2.addLayout(self.horizontalLayout4)
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
        self.verticalLayout_5.addWidget(self.groupBox_2)
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
        self.spinBox_2.setStyleSheet("font: 9pt \"Arial\";")
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout_7.addWidget(self.spinBox_2)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_5.addWidget(self.groupBox_5)
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
        self.graphicsView = QtWidgets.QLabel(self.Box_5)
        self.graphicsView.setMaximumSize(QtCore.QSize(self.desktop.width(), self.desktop.height()))
        self.graphicsView.setContentsMargins(2, 2, 2, 2)
        self.graphicsView.setScaledContents(True)
        self.width = self.graphicsView.width()
        self.height = self.graphicsView.height()
        self.graphicsView.setObjectName("graphicsView")
        self.COVER = 'images/initiate.png'
        self.img1 = cv2.imread(self.COVER)
        image1 = cv2.pyrDown(self.img1)
        y, x = image1.shape[:-1]
        self.cvimg1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 把opencv 默认BGR转为通用的RGB
        self.frame1 = QImage(self.cvimg1, x, y, QImage.Format_RGB888)
        self.pix = QPixmap.fromImage(self.frame1)
        self.scene = QGraphicsScene()  # 创建画布
        self.graphicsView.setPixmap(self.pix)
        self.graphicsView.show()
        self.scene.addPixmap(self.pix)
        self.horizontalLayout_6.addWidget(self.graphicsView)
        self.horizontalLayout_6.setStretch(0, 1)
        self.horizontalLayout_6.setStretch(1, 7)
        self.verticalLayout_6.addWidget(self.Box_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 687, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.retranslateUi(MainWindow)
        self.connect()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
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
        self.groupBox_5.setTitle(_translate("MainWindow", "脱离范围检测"))
        self.checkBox_4.setText(_translate("MainWindow", "脱离范围检测"))
        self.label_3.setText(_translate("MainWindow", "脱离时间(帧):"))
        self.spinBox.setRange(1, 180)
        self.spinBox1.setRange(1, 180)
        self.spinBox_2.setRange(100, 200)
        self.spinBox2.setRange(100, 1800)
        self.spinBox.setSingleStep(10)
        self.spinBox1.setSingleStep(1)
        self.spinBox2.setSingleStep(50)
        self.spinBox_2.setSingleStep(15)
        self.groupBox_4.setTitle(_translate("MainWindow", "状态输出"))
        self.checkBox.setChecked(True)
        self.checkBox3.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox_4.setChecked(True)

    def connect(self):
        self.mtt = []
        self.pushButton_2.clicked.connect(self.camera_on)
        self.pushButton_4.clicked.connect(self.off)
        # 硬编码使用本地摄像头
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        # 闭眼检测帧数，闭眼达到该帧数则判断为睡觉
        self.AR_CONSEC_FRAMES_check = 60
        # 脱离检测范围帧数，脱离摄像头达到该帧数判断为脱岗
        self.OUT_AR_CONSEC_FRAMES_check = 90
        self.spinBox.valueChanged.connect(self.AR_CONSEC_FRAMES)
        self.spinBox_2.valueChanged.connect(self.OUT_AR_CONSEC_FRAMES)
        self.spinBox1.valueChanged.connect(self.Number_Of_Yawns_Judged_As_Fatigue_2)
        self.spinBox2.valueChanged.connect(self.NOYJAF_Time_2)
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.24
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        # 打哈欠长宽比
        self.MAR_THRESH = 0.6
        self.MOUTH_AR_CONSEC_FRAMES = 30
        self.time_reduce = 0
        self.spinBox_2.setValue(self.OUT_AR_CONSEC_FRAMES_check)
        self.spinBox.setValue(self.AR_CONSEC_FRAMES_check)
        """计数"""
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
        self.timeOfTheLastOfYawns = datetime.datetime(2022, 12, 31)
        self.timeOfTheFirstOfYawns = datetime.datetime(2022, 12, 31)

    def __del__(self):
        pass

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
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
        self.textBrowser.append(u"加载模型成功!!!\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)

    def isOpened(self):
        self.CAMERA_STYLE = True
        self.textBrowser.append(u"打开本地摄像头成功!!!\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)

    def notOpen(self):
        self.textBrowser.append(u"本地摄像头打开失败!!!\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.scene.clear()
        self.scene.addPixmap(self.pix)

    def putText(self, im_rd, mar):
        if self.checkBox3.isChecked() == True:
            cv2.putText(im_rd, "mCOUNTER: {}".format(self.mCOUNTER), (5, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
            cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
            cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1)
            cv2.putText(im_rd, "FatigueTime: {:.2f}".format(self.time_reduce), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 0), 1)

    def img(self, im_rd, faces, ear):
        if self.checkBox3.isChecked() == True:
            cv2.putText(im_rd, "FACES: {}".format(len(faces)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
            cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (70, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
            cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)

    def _learning_face(self, event):
        self.spredictor()
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.cap = cv2.VideoCapture(0)  # 硬编码使用本地摄像头
        if self.cap.isOpened():
            self.isOpened()
        else:
            self.notOpen()
        self.time_start = time.perf_counter()
        while self.cap.isOpened():
            flag, im_rd = self.cap.read()
            if not flag:
                continue
            im_rd = imutils.resize(im_rd, 240)
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            if len(faces) != 0:
                for k, d in enumerate(faces):
                    import _thread
                    if self.checkBox3.isChecked() == True:
                        _thread.start_new_thread(cv2.rectangle,
                                                 (im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1))
                    shape = self.predictor(im_rd, d)
                    if self.checkBox3.isChecked() == True:
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    shape = face_utils.shape_to_np(shape)
                    ll = (shape[67, 0], shape[67, 1])
                    oo = (shape[63, 0], shape[63, 1])
                    self.ooo = -(ll[0] - oo[0]) + ll[1] - oo[1]
                    if self.checkBox.isChecked():
                        mouth = shape[mStart:mEnd]
                        mar = self.mouth_aspect_ratio(mouth)
                        mouthHull = cv2.convexHull(mouth)
                        if self.checkBox3.isChecked() == True:
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
                                self.textBrowser.append("打哈欠")
                            if self.checkBox3.isChecked() == True:
                                cv2.putText(im_rd, "Yawming", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.putText(im_rd, mar)
                        if self.time_reduce >= 60 and self.ooo < 20:
                            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                    if self.checkBox_2.isChecked() == True:
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        import _thread
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        if self.checkBox3.isChecked() == True:
                            _thread.start_new_thread(cv2.drawContours, (im_rd, [leftEyeHull], -1, (0, 255, 0), 1,))
                            _thread.start_new_thread(cv2.drawContours, (im_rd, [rightEyeHull], -1, (0, 255, 0), 1,))
                        self.time_end = time.perf_counter()
                        self.time_reduce = self.time_end - self.time_start
                        if ear < self.EYE_AR_THRESH:
                            self.COUNTER += 1
                        if ear > self.EYE_AR_THRESH:
                            self.COUNTER = 0
                        else:
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                                self.shutEye = True
                                self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                        if self.mTOTAL >= self.Number_Of_Yawns_Judged_As_Fatigue and (0 < ((self.timeOfTheLastOfYawns - self.timeOfTheFirstOfYawns).seconds) < self.NOYJAF_Time):
                            self.ifTired = True
                        self.img(im_rd, faces, ear)
            else:
                self.oCOUNTER += 1
                if self.checkBox3.isChecked() == True:
                    cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.textBrowser.append(time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime()) + u"脱离识别范围!!!\n")
                    self.ifNoFace = True
                    t = threading.Thread(target=self.noFace)
                    t.start()
                    self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                    self.oCOUNTER = 0
            if self.shutEye:
                self.textBrowser.append("睡觉")
                t = threading.Thread(target=self.sleep)
                t.start()
                self.shutEye = False
                self.COUNTER = 0
            if self.ifTired:
                self.TOTAL = 0
                self.mTOTAL = 0
                self.hTOTAL = 0
                self.eToTAL = 0
                self.textBrowser.append("疲劳")
                t = threading.Thread(target=self.tired)
                t.start()
                self.ifTired = False
                self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
            image1 = cv2.pyrUp(im_rd)
            y, x = image1.shape[:-1]
            self.width = self.centralwidget.width()
            self.height = self.centralwidget.height()
            s1 = (self.width - self.width * 1 / 7) > x
            s3 = self.height > y
            if s1 & False:
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

    def camera_on(self, event):
        import _thread
        _thread.start_new_thread(self._learning_face, (event,))

    def AR_CONSEC_FRAMES(self, event):
        self.textBrowser.append(u"设置疲劳间隔为:\t" + str(self.spinBox.value()) + "秒\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.AR_CONSEC_FRAMES_check = int(self.spinBox.value())

    def OUT_AR_CONSEC_FRAMES(self, event):
        self.textBrowser.append(u"设置脱离识别间隔为:\t" + str(self.spinBox_2.value()) + "秒\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.OUT_AR_CONSEC_FRAMES_check = int(self.spinBox_2.value())

    def Number_Of_Yawns_Judged_As_Fatigue_2(self, event):
        self.textBrowser.append(u"设置哈欠次数为:\t" + str(self.spinBox1.value()) + "\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.Number_Of_Yawns_Judged_As_Fatigue = int(self.spinBox1.value())

    def NOYJAF_Time_2(self, event):
        self.textBrowser.append(u"设置间隔时间为:\t" + str(self.spinBox2.value()) + "\n")
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
        self.NOYJAF_Time = int(self.spinBox2.value())

    def off(self):
        if self.CAMERA_STYLE:
            self.CAMERA_STYLE = False
            self.cap.release()
            self.scene.addPixmap(self.pix)

import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())