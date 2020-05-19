import sys
import os
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import re
import random
import time
import src.params as params
import src.information as information

from src.face_detection import *
from src.face_recognition import *
from src.face_landmark import *
from src.pose_estimate import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

EYE_COUNTER = 0
MOUTH_COUNTER = 0
COUNT = 0
EYE_ALARM_ON = False
MOUTH_ALARM_ON = False

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0

        self.mtcnn = MTCNN()

        self.predictor = dlib.shape_predictor(params.LANDMARK_MODEL_PATH)

        self.detector = tf.lite.Interpreter(params.DETECT_MODEL_PATH)
        #self.labels = load_labels(os.path.join('model', 'coco_labels.txt'))
        self.detector.allocate_tensors()
        self.input_details = self.detector.get_input_details()
        self.output_details = self.detector.get_output_details()
        
        self.posenet = tf.lite.Interpreter(params.POSE_MODEL_PATH)
        self.posenet.allocate_tensors()
        self.p_input_details = self.posenet.get_input_details()
        self.p_output_details = self.posenet.get_output_details()
        
        self.arcface = tf.lite.Interpreter(params.FACE_MODEL_PATH)
        self.arcface.allocate_tensors()
        self.a_input_details = self.arcface.get_input_details()
        self.a_output_details = self.arcface.get_output_details()

        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.out = cv2.VideoWriter('out.avi', fourcc, 20.0, (640, 480))

        

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_fun_button = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # QVBoxLayout类垂直地摆放小部件

        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_close = QtWidgets.QPushButton(u'退出')

        # button颜色修改
        button_color = [self.button_open_camera, self.button_close]
        for i in range(2):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{border_radius:10px}"
                                           "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        # move()方法是移动窗口在屏幕上的位置到x = 500，y = 500的位置上
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(641, 481)
        self.label_show_camera.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'测试界面')


    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.button_close.clicked.connect(self.close)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.button_open_camera.setText(u'关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'打开相机')

    def show_camera(self):
        global EYE_COUNTER
        global MOUTH_COUNTER
        global EYE_ALARM_ON
        global MOUTH_ALARM_ON
        global COUNT
        #time_start=time.time()

        flag, self.image = self.cap.read()

        boxes,points = self.mtcnn.predict(self.image)
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        if len(boxes) == 0:
            COUNT = COUNT + 1
        else:
            COUNT = 0
        if COUNT >= 10:
            cv2.putText(self.image, u'Warning!', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for box in boxes:
            score = box[4]
            if score < params.THRESHOLD:
                break

            box = box[:4] +0.5
            box = box.astype(np.int32)
            cut = self.image[box[1]:box[3], box[0]:box[2]]
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0,0,255))  

            if points.shape == (10,1):
                cv2.rectangle(self.image, (points[0] - int((box[3] - box[1]) / 8), points[5] - int((box[3] - box[1]) / 8)),
                                         (points[0] + int((box[3] - box[1]) / 8), points[5] + int((box[3] - box[1]) / 8)), (0,0,255))  
                cv2.rectangle(self.image, (points[1] - int((box[3] - box[1]) / 8), points[6] - int((box[3] - box[1]) / 8)),
                                         (points[1] + int((box[3] - box[1]) / 8), points[6] + int((box[3] - box[1]) / 8)), (0,0,255)) 
                cv2.rectangle(self.image, (points[3], points[8] - int((box[3] - box[1]) / 8)),
                                         (points[4], points[9] + int((box[3] - box[1]) / 8)), (0,0,255)) 

            rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            

        #time_end=time.time()

        #cv2.putText(self.image, u'FPS:{:.0f}'.format(1 / (time_end-time_start)), (500, 100),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        show = cv2.resize(self.image, (640, 480))
        self.out.write(show)
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
               
        

        def closeEvent(self, event):
            ok = QtWidgets.QPushButton()
            cancel = QtWidgets.QPushButton()
            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
            msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cancel.setText(u'取消')
            if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
                event.ignore()
            else:
                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()
                self.out.release()
                event.accept()


if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
    
    
