import sys
import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QPixmap
import dlib
import face_recognition
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

class AnotherWindow(QWidget):
    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath
        layout = QVBoxLayout()
        self.label = QLabel('new window', self)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.mypixmap = QPixmap()
        self.mypixmap.load(self.imagePath)
        self.mypixmap = self.mypixmap.scaled(850, 620, Qt.KeepAspectRatio)

        if self.mypixmap.isNull():
            print('Image not found')
            return
        else:
            self.label.setPixmap(self.mypixmap)


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  # 初始化攝像頭定時器
        self.timer_camera_face = QtCore.QTimer() # 初始化人臉檢測定時器 
        self.timer_camera_landmark = QtCore.QTimer() # 初始化關鍵點檢測定時器
        self.timer_camera_expression = QtCore.QTimer() # 初始化情緒定時器
        self.cap = cv2.VideoCapture()  # 初始化攝像頭
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.x = 0
        self.count = 0
        # 人臉檢測
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        # 情緒檢測器
        self.expression_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        self.expression_model = load_model('epoch_30.hdf5')
        self.EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

        self.window1 = AnotherWindow('./camera.png')
        self.window2 = AnotherWindow('./face.png')
        self.window3 = AnotherWindow('./landmark.png')
        self.window4 = AnotherWindow('./expression.png')

        
    def set_ui(self):
        '''
        建立window --> window = QWidget() 
        建立button --> button = QtPushButton()
        建立layout --> layout = QHBoxLayout(window)
        網layout添加widget --> layout.addWidget(button)
        '''
        # self.__layout_main = QtWidgets.QHBoxLayout()  # QHBoxLayout水平佈局，按照從左到右的順序排列
        # self.__layout_buttons = QtWidgets.QHBoxLayout()
        
        # main vertical layout
        self.__layout_main = QtWidgets.QVBoxLayout()
        self.__layout_buttons = QtWidgets.QHBoxLayout()
        self.__layout_buttons2 = QtWidgets.QHBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'打開攝像頭')
        self.button_open_face = QtWidgets.QPushButton(u'打開人臉檢測')
        self.button_open_landmark = QtWidgets.QPushButton(u'打開人臉關鍵點檢測')
        self.button_open_expression = QtWidgets.QPushButton(u'打開人臉情緒檢測')
        self.button_close_camera = QtWidgets.QPushButton(u'關閉程序')

        self.button_show_camera_code = QtWidgets.QPushButton(u'攝像頭程式')
        self.button_show_face_code = QtWidgets.QPushButton(u'人臉檢測程式')
        self.button_show_landmark_code = QtWidgets.QPushButton(u'關鍵點程式')
        self.button_show_expression_code= QtWidgets.QPushButton(u'表情檢測程式')

        # button颜色修改
        button_color = [self.button_open_camera, self.button_close_camera, self.button_open_face, self.button_open_landmark, self.button_show_camera_code, \
                        self.button_show_face_code, self.button_show_landmark_code, self.button_show_expression_code, self.button_open_expression
                    ] 
        button_color_count = len(button_color)
        for i in range(button_color_count):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{padding:2px 4px}")
        # 設置button最小高度
        for button in button_color:
            button.setMinimumHeight(50)
        # self.button_open_camera.setMinimumHeight(50)
        # self.button_close_camera.setMinimumHeight(50)
        # self.button_open_face.setMinimumHeight(50)
        # self.button_open_landmark.setMinimumHeight(50)

        # move(x, y) --> 移動介面到指定位置。 (0, 0)位於最左上角, 往右和往下為正
        self.move(350, 150)

        # 右側顯示攝像頭的panel
        self.show_camera_panel = QtWidgets.QLabel()
        self.show_camera_panel.setFixedSize(800, 600)    # 整個camera panel的尺寸大小        
 
        # setAutoFillBackground要和palette一起用
        # 這裏用來表示顯示camera panel
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        self.show_camera_panel.setPalette(palette)
        self.show_camera_panel.setAutoFillBackground(True)
        # 第一行button添加到button layout中
        self.__layout_buttons.addWidget(self.button_open_camera)
        self.__layout_buttons.addWidget(self.button_open_face)
        self.__layout_buttons.addWidget(self.button_open_landmark)
        self.__layout_buttons.addWidget(self.button_open_expression)
        self.__layout_buttons.addWidget(self.button_close_camera)
        # 第二行button添加到第二行的button layout 中
        self.__layout_buttons2.addWidget(self.button_show_camera_code)
        self.__layout_buttons2.addWidget(self.button_show_face_code)
        self.__layout_buttons2.addWidget(self.button_show_landmark_code)
        self.__layout_buttons2.addWidget(self.button_show_expression_code)
        # 把button layout添加到main layout上
        self.__layout_main.addLayout(self.__layout_buttons)
        self.__layout_main.addLayout(self.__layout_buttons2)
        self.__layout_main.addWidget(self.show_camera_panel)
       
        self.setWindowTitle(u'主程序畫面')
        self.setLayout(self.__layout_main)
        # self.label_move.raise_()      # raise_()把該widget置於最上層

    def slot_init(self):
        # 建立打開攝像頭連接
        self.button_open_camera.clicked.connect(lambda: self.button_click(btn_num = 0))
        self.timer_camera.timeout.connect(self.show_camera)
        # 建立人臉檢測連接
        self.button_open_face.clicked.connect(lambda: self.button_click(btn_num = 1))
        self.timer_camera_face.timeout.connect(self.face_detector)
        # 建立關鍵點檢測連接
        self.button_open_landmark.clicked.connect(lambda: self.button_click(btn_num = 2))
        self.timer_camera_landmark.timeout.connect(self.face_landmark)       
        # 建立情緒連接
        self.button_open_expression.clicked.connect(lambda: self.button_click(btn_num = 3))
        self.timer_camera_expression.timeout.connect(self.facial_expression)       
        # 建立關閉程序
        self.button_close_camera.clicked.connect(self.close) 

        # 點擊打開攝像頭程式
        self.button_show_camera_code.clicked.connect(self.show_camera_code)
        # 點擊打開人臉檢測程式
        self.button_show_face_code.clicked.connect(self.show_facial_code)
        # 點擊打開人臉關鍵點檢測程式
        self.button_show_landmark_code.clicked.connect(self.show_landmark_code)
        # 點擊打開人臉情緒檢測程式 
        self.button_show_expression_code.clicked.connect(self.show_expression_code)
    
    def button_click(self, btn_num):
        click_buttons = [self.button_open_camera, self.button_open_face, self.button_open_landmark, self.button_open_expression]
        timers = [self.timer_camera, self.timer_camera_face, self.timer_camera_landmark, self.timer_camera_expression]
        texts = [u'打開攝像頭', u'打開人臉檢測', u'打開人臉關鍵點檢測', u'打開人臉情緒檢測']
        if timers[btn_num].isActive() == False:
            flag = self.cap.open(self.CAM_NUM)   # 筆記本電腦的攝像頭通常CAM_NUM = 0
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'注意', u'請檢測外部攝像頭與電腦是否連接正常',
                                                            buttons=QtWidgets.QMessageBox.Ok,
                                                            defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # 10代表每10ms觸發一次
                timers[btn_num].start(10)
                click_buttons[btn_num].setText(u'關閉攝像頭')
        else:
            timers[btn_num].stop()
            self.cap.release()
            self.show_camera_panel.clear()
            click_buttons[btn_num].setText(texts[btn_num])


    def show_camera(self):
        _, self.frame = self.cap.read()     # --> frame: (height, width, channel)
        show = cv2.resize(self.frame, (800, 600))  # 顯示在camera panel中的圖片大小, 想要的尺寸(width, height)
         # opencv默認的BGR要轉換成RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # --> (600, 800, 3)
        # showImage return 一個object.    QtGui.QImage(data, width, height, format)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.show_camera_panel.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def face_detector(self):
        # self.cap.open(self.CAM_NUM)
        _, self.frame = self.cap.read()     # --> frame: (height, width, channel)
        show = cv2.resize(self.frame, (800, 600))  # 顯示在camera panel中的圖片大小, 想要的尺寸(width, height)
         # opencv默認的BGR要轉換成RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # --> (600, 800, 3)
        # showImage return 一個object.    QtGui.QImage(data, width, height, format)
        boxes = face_recognition.face_locations(show, model = 'hog')
        for (top, right, bottom, left) in boxes:
            cv2.rectangle(show, (left, top), (right + 17, bottom + 17), (0, 255, 0), 2)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.show_camera_panel.setPixmap(QtGui.QPixmap.fromImage(showImage))

    
    def face_landmark(self):
        _, self.frame = self.cap.read()
        show = cv2.resize(self.frame, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        points = self.detector(show, 1)
        for point in points:
            shape = self.predictor(show, point)
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape = shape_np

            for i, (x, y) in enumerate(shape):
                cv2.circle(show, (x, y), 2, (0, 255, 0), -1)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.show_camera_panel.setPixmap(QtGui.QPixmap.fromImage(showImage)) 

    def facial_expression(self): 
        _, self.frame = self.cap.read()
        show = cv2.resize(self.frame, (800, 600))
        show_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show_gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        # canvas = np.zeros((220, 300, 3), dtype = 'uint8') # 用來畫情緒可能的分佈
        # frameClone = show.copy()
        rects = self.expression_detector.detectMultiScale(show_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
										flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) > 0:
            rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0] # 最大那张脸的boundary
            (X, Y, W, H) = rect

            roi = show_gray[Y:Y + H, X:X + W]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 把roi输入model来预测
            preds = self.expression_model.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]

        #     # for (_, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
        #     #     text = "{}: {:.2f}%".format(emotion, prob * 100)

        #         # w = int(prob * 300)
        #         # cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        #         # cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            cv2.putText(show_rgb, label, (X, Y - 17), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(show_rgb, (X, Y), (X + W, Y + H),	(0, 255, 0), 2)
            
            showImage = QtGui.QImage(show_rgb.data, show_rgb.shape[1], show_rgb.shape[0], QtGui.QImage.Format_RGB888)
            self.show_camera_panel.setPixmap(QtGui.QPixmap.fromImage(showImage)) 

            
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'關閉', u'確定關閉程序？')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'確定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    def show_camera_code(self, checked):
        if self.window1.isVisible():
            self.window1.hide()
        else:
            self.window1.show()

    def show_facial_code(self, checked):    
        if self.window2.isVisible():
            self.window2.hide()
        else:
            self.window2.show() 
        
    def show_landmark_code(self, checked):
        if self.window3.isVisible():
            self.window3.hide()
        else:
            self.window3.show()

    def show_expression_code(self, checked):
        if self.window4.isVisible():
            self.window4.hide()
        else:
            self.window4.show()


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(App.exec_())