from  datetime import datetime
import cv2
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtCore, QtGui, QtWidgets
import dlib
from keras.models import load_model

class FacialRecognition:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

        self.expression_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
        self.expression_model = load_model('epoch_30.hdf5')
        self.EMOTIONS = ["生氣", "驚嚇", "開心", "傷心", "驚訝", "普通"]
        self.EMOJI = ['emojis/angry_emoji.png', 'emojis/scared_emoji.png', 'emojis/happy_emoji.png', 'emojis/sad_emoji.png', 'emojis/surprised_emoji.png', 'emojis/neutral_emoji.png']
        
    def facial_expression(self, filename = None): 
        now = datetime.now()
        curr = now.strftime("%d-%m-%y-%H-%M-%S")

        frame = cv2.imread(filename)
        show = cv2.resize(frame, (500, 500))
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
            showImage.save(f'./captured/expressions/photo_expression_{curr}.png')