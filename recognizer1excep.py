# -*- coding: utf-8 -*-
import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2,os
import numpy as np

class CameraApp(App):

    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.img = Image()
        Clock.schedule_interval(self.update, 1/30)
        self.faceCascade = cv2.CascadeClassifier("Cascades\\haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainner\\trainner.yml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        return self.img

    def update(self, dt):
        ret, im = self.capture.read()

        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=self.faceCascade.detectMultiScale(gray, 1.2, 5)
        for(x,y,w,h) in faces:
            nbr_predicted, conf = self.recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(0,255,0),2)        
            if(nbr_predicted==15 or nbr_predicted==26 or nbr_predicted==8):   
                nbr_predicted='Anton'
            if nbr_predicted!='Anton':
                nbr_predicted='Unknown'
            cv2.putText(im,str(nbr_predicted), (x,y+h),self.font, 1,(0,255,0),4) #Draw the text
            
            buf1 = cv2.flip(im, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture                    
            cv2.waitKey(10)
    
      
     
        

if __name__ == '__main__':
    CameraApp().run()

