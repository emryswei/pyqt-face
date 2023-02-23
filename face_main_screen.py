import sys
import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import dlib
import face_recognition
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ShowMainWindow(QtWidgets.QWidget):
    switch_window_func = QtCore.pyqtSignal()
    switch_window_code = QtCore.pyqtSignal()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setWindowTitle(u'主程式畫面')
        self.layout_main = QtWidgets.QVBoxLayout()
        self.button_func_show = QtWidgets.QPushButton(u'進入演示模式')
        self.button_code_show = QtWidgets.QPushButton(u'進入拍照試玩模式')
        self.buttons = [self.button_func_show, self.button_code_show]
        # 設置window的開始位置和尺寸
        self.setGeometry(350, 150, 600, 600)
        # 添加button樣式為expanding自適應
        self.button_adaptive = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.button_func_show.setSizePolicy(self.button_adaptive)
        self.button_code_show.setSizePolicy(self.button_adaptive)
        # 設置button text樣式
        for button in self.buttons:
            button.setFont(QFont('Times', 48))
        # 設置button樣式
        for button in self.buttons:
            button.setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(229,255,204)}"
                                           "QpushButton{border:5px}"
                                           "QPushButton{padding:2px 4px}")
        # 在layout_main上添加button
        self.layout_main.addWidget(self.button_func_show)
        self.layout_main.addSpacing(10)
        self.layout_main.addWidget(self.button_code_show)
        # 添加button click事件
        self.button_func_show.clicked.connect(self.switch_func)
        self.button_code_show.clicked.connect(self.switch_code)
        # 應用上面設置的layout
        self.setLayout(self.layout_main)

    def switch_func(self):
        self.switch_window_func.emit()

    def switch_code(self):
        self.switch_window_code.emit()

class ShowFunctionWindow(QtWidgets.QWidget):
    pass

class ShowCodeWindow(QtWidgets.QWidget):
    pass

class Controller:
    def __init__(self):
        pass

    def show_main(self):
        self.main_window = ShowMainWindow()
        self.main_window.switch_window_func.connect(self.show_function)
        # 暫時用code_window測試
        self.main_window.switch_window_code.connect(self.show_test)
        self.main_window.show()

    def show_function(self):
        self.func_window = Ui_MainWindow()
        self.main_window.close()
        self.func_window.show()

    def show_code(self):
        pass

    def show_test(self):
        self.test = TestCapture()
        self.main_window.close()
        self.test.show()
        self.test.button_back.clicked.connect(lambda: {self.test.close(), self.main_window.show()})


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
        # self.mypixmap.setDevicePixelRatio(1)
        self.mypixmap = self.mypixmap.scaled(1000, 700, Qt.KeepAspectRatio)

        if self.mypixmap.isNull():
            print('Image not found')
            return
        else:
            self.label.setPixmap(self.mypixmap)
            # self.resize(850, 620)


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
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

        # 情緒檢測器
        self.expression_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        self.expression_model = load_model('epoch_30.hdf5')
        self.EMOTIONS = ["生氣", "驚嚇", "開心", "傷心", "驚訝", "普通"]
        self.EMOJI = ['emojis/angry_emoji.png', 'emojis/scared_emoji.png', 'emojis/happy_emoji.png', 'emojis/sad_emoji.png', 'emojis/surprised_emoji.png', 'emojis/neutral_emoji.png']
        self.window1 = AnotherWindow('codes/camera.png')
        self.window2 = AnotherWindow('codes/face.png')
        self.window3 = AnotherWindow('codes/landmark.png')
        self.window4 = AnotherWindow('codes/expression.png')


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
        self.button_show_expression_code= QtWidgets.QPushButton(u'情緒檢測程式')

        # button颜色修改
        button_color = [self.button_open_camera, self.button_close_camera, self.button_open_face, self.button_open_landmark, self.button_show_camera_code, \
                        self.button_show_face_code, self.button_show_landmark_code, self.button_show_expression_code, self.button_open_expression,\
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
        _, self.frame = self.cap.read()     # --> frame: (height, width, channel)
        show = cv2.resize(self.frame, (800, 600))  # 顯示在camera panel中的圖片大小, 想要的尺寸(width, height)
         # opencv默認的BGR要轉換成RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # --> (600, 800, 3)
        # showImage return 一個object.    QtGui.QImage(data, width, height, format)
        # boxes = face_recognition.face_locations(show, model = 'hog')
        # for (top, right, bottom, left) in boxes:
        #     cv2.rectangle(show, (left, top), (right + 17, bottom + 17), (0, 255, 0), 2)

        h, w = show.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(show, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(show, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # faceRects = self.detector(show, 1)
        # for _, face in enumerate(faceRects):
        #     cv2.rectangle(show, (face.left(), face.top()), (face.right() + 17, face.bottom() + 17), (0, 255, 0), 2)
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
        # rects = self.expression_detector.detectMultiScale(show_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
		# 								flags=cv2.CASCADE_SCALE_IMAGE)
        h, w = show.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(show, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # cv2.rectangle(show, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # rects = face_recognition.face_locations(show_rgb, model = 'hog')
        # for (top, right, bottom, left) in rects:
            # cv2.rectangle(show, (left, top), (right + 17, bottom + 17), (0, 255, 0), 2)

        # if len(rects) > 0:
        #     rect = sorted(rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            # roi = show_gray[top: bottom, left: right]
            # roi = show_gray[Y:Y + H, X:X + W]
            X, Y, W, H = startX - 5, startY, (endX - startX + 5), (endY - startY)
            roi = show_gray[Y:Y + H, X:X + W]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 把roi输入model来预测
            preds = self.expression_model.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]
            emoji = self.EMOJI[preds.argmax()]
            
            img = Image.fromarray(show)
            draw = ImageDraw.Draw(img)
            fontText = ImageFont.truetype('font/simsun.ttc', size = 30, encoding = 'utf-8')
            draw.text((X, Y - 36), label, (0, 255, 0), stroke_width = 1, font = fontText)
            img = np.array(img)
            show_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.putText(show_rgb, label, (X, Y - 17), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(show_rgb, (X, Y), (X + W, Y + H + 18), (0, 255, 0), 2)
            
            emoji_image = cv2.imread(emoji, -1)
            emoji_image = cv2.cvtColor(emoji_image, cv2.COLOR_BGR2RGBA)
            emoji_image = cv2.resize(emoji_image, (100, 100))
            alpha_emoji = emoji_image[:, :, 3] / 255.0
            alpha_show_rgb = 1.0 - alpha_emoji

            emoji_start  = X - 110
            emoji_end = X - 10
            if X - 110 < 0:
                emoji_start = X + W + 10
                emoji_end = X + W + 110
            for c in range(0, 3):
                show_rgb[Y:(Y+100), emoji_start:emoji_end, c] = (alpha_emoji * emoji_image[:, :, c] + alpha_show_rgb * show_rgb[Y:(Y+100), emoji_start:emoji_end, c])
            
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


class TestCapture(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TestCapture, self).__init__(parent) 

        self.timer_camera = QtCore.QTimer()  # 初始化攝像頭定時器
        self.cap = cv2.VideoCapture()  # 初始化攝像頭
        self.set_ui()
        self.slot_init()
        self.show_image = None

    def set_ui(self):
        self.main_layout = QVBoxLayout()
        self.utilities_layout = QHBoxLayout()
        self.button_open_camera = QPushButton(u'攝像頭功能')
        self.button_cap = QPushButton(u"拍照")
        self.button_open_img = QPushButton(u"查看照片")
        self.button_back = QPushButton(u"回到主頁")
        # button颜色修改
        self.buttons = [self.button_open_camera, self.button_cap, self.button_open_img, self.button_back]
        for button in self.buttons:
            button.setStyleSheet("QPushButton{color:black}"
                                "QPushButton:hover{color:red}"
                                "QPushButton{background-color:rgb(78,255,255)}"
                                "QpushButton{border:2px}"
                                "QPushButton{padding:2px 4px}")
        # 設置button最小高度
        for button in self.buttons:
            button.setMinimumHeight(50)
        # move(x, y) --> 移動介面到指定位置。 (0, 0)位於最左上角, 往右和往下為正
        self.move(350, 150)

        # 下方顯示攝像頭的panel
        self.show_camera_panel = QtWidgets.QLabel()
        self.show_camera_panel.setFixedSize(800, 600)    # 整個camera panel的尺寸大小        
 
        # setAutoFillBackground要和palette一起用 | palette: 調色板
        # 這裏用來表示顯示camera panel  
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        self.show_camera_panel.setPalette(palette)
        self.show_camera_panel.setAutoFillBackground(True)

        # 把button layout添加到main layout上
        self.main_layout.addWidget(self.show_camera_panel)
        self.main_layout.addLayout(self.utilities_layout)
        self.utilities_layout.addWidget(self.button_open_camera)
        self.utilities_layout.addWidget(self.button_cap)
        self.utilities_layout.addWidget(self.button_open_img)
        self.main_layout.addWidget(self.button_back)
        self.setWindowTitle(u'攝像頭捕捉畫面')
        self.setLayout(self.layout)
        
    def slot_init(self):
        # 建立打開攝像頭連接    
        self.button_open_camera.clicked.connect(self.button_click)      
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_cap.clicked.connect(self.cap_image)
        self.button_open_img.clicked.connect(self.open_directory)

    # 開關攝像頭
    def button_click(self):
        timers = self.timer_camera
        if timers.isActive() == False:
            flag = self.cap.open(0)   # 筆記本電腦的攝像頭通常CAM_NUM = 0
            if flag == False:
                msg = QtWidgets.QMessageBox.Warning(self, u'注意', u'請檢測外部攝像頭與電腦是否連接正常', buttons=QtWidgets.QMessageBox.Ok,
                                                            defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # 10代表每10ms觸發一次
                timers.start(10)
        else:
            timers.stop()
            self.cap.release()
            self.show_camera_panel.clear() 
    # 拍照
    def cap_image(self):
        self.show_image.save('./captured/photo.png')
        msg = QMessageBox(QMessageBox.Warning, "提示", '拍照成功')
        msg.exec_()
    # 打開照片文件夾
    def open_directory(self):
        self.captured = CapturedImageSelect()
        self.captured.show()
    # 打開攝像頭
    def show_camera(self):
        _, self.frame = self.cap.read()     # --> frame: (height, width, channel)
        show = cv2.resize(self.frame, (800, 600))  # 顯示在camera panel中的圖片大小, 想要的尺寸(width, height)
         # opencv默認的BGR要轉換成RGB
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # --> (600, 800, 3)
        # showImage return 一個object.    QtGui.QImage(data, width, height, format)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.show_image = showImage
        self.show_camera_panel.setPixmap(QtGui.QPixmap.fromImage(showImage))


class CapturedImageSelect(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()  

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        # 情緒檢測器
        self.expression_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        self.expression_model = load_model('epoch_30.hdf5')
        self.EMOTIONS = ["生氣", "驚嚇", "開心", "傷心", "驚訝", "普通"]
        self.EMOJI = ['emojis/angry_emoji.png', 'emojis/scared_emoji.png', 'emojis/happy_emoji.png', 'emojis/sad_emoji.png', 'emojis/surprised_emoji.png', 'emojis/neutral_emoji.png']
        
        self.setWindowTitle("照片文件夾")
        self.move(200,200)

        self.layout = QVBoxLayout()
        self.button = QPushButton('選擇照片')
        self.button.setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{padding:2px 4px}")
        self.button.clicked.connect(self.open_image)
        self.panel = QLabel()
        self.panel.setFixedSize(500, 500)    # 整個camera panel的尺寸大小
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.black)
        self.panel.setPalette(palette)
        self.panel.setAutoFillBackground(True)

        self.layout.addWidget(self.button)
        self.layout.addWidget(self.panel)
        self.setLayout(self.layout)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打開文件夾', './captured', "Image files (*.jpg *.png)")
        self.img = cv2.imread(fname)
        self.img = cv2.resize(self.img, (500, 500))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], QtGui.QImage.Format_RGB888)
        self.panel.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.facial_expression(fname)

    def facial_expression(self, filename = None): 
        self.frame = cv2.imread(filename)
        show = cv2.resize(self.frame, (500, 500))
        show_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show_gray = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)

        h, w = show.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(show, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            X, Y, W, H = startX - 5, startY, (endX - startX + 5), (endY - startY)
            roi = show_gray[Y:Y + H, X:X + W]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.expression_model.predict(roi)[0]
            label = self.EMOTIONS[preds.argmax()]
            emoji = self.EMOJI[preds.argmax()]
            
            img = Image.fromarray(show)
            draw = ImageDraw.Draw(img)
            fontText = ImageFont.truetype('font/simsun.ttc', size = 30, encoding = 'utf-8')
            draw.text((X, Y - 36), label, (0, 255, 0), stroke_width = 1, font = fontText)
            img = np.array(img)
            show_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.rectangle(show_rgb, (X, Y), (X + W, Y + H + 18), (0, 255, 0), 2)
            
            emoji_image = cv2.imread(emoji, -1)
            emoji_image = cv2.cvtColor(emoji_image, cv2.COLOR_BGR2RGBA)
            emoji_image = cv2.resize(emoji_image, (100, 100))
            alpha_emoji = emoji_image[:, :, 3] / 255.0
            alpha_show_rgb = 1.0 - alpha_emoji

            emoji_start  = X - 110
            emoji_end = X - 10
            if X - 110 < 0:
                emoji_start = X + W + 10
                emoji_end = X + W + 110
            for c in range(0, 3):
                show_rgb[Y:(Y+100), emoji_start:emoji_end, c] = (alpha_emoji * emoji_image[:, :, c] + alpha_show_rgb * show_rgb[Y:(Y+100), emoji_start:emoji_end, c])
            
            showImage = QtGui.QImage(show_rgb.data, show_rgb.shape[1], show_rgb.shape[0], QtGui.QImage.Format_RGB888)
            self.panel.setPixmap(QtGui.QPixmap.fromImage(showImage)) 
            showImage.save('./captured/photo_expression.png')



if __name__ == '__main__':
    App = QApplication(sys.argv)
    # window = Ui_MainWindow()
    # window = ShowMainWindow()
    # window.show()
    controller = Controller()
    controller.show_main()
    sys.exit(App.exec_())