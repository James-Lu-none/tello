from djitellopy.tello import Tello 
from threading import Thread
import cv2
import mediapipe as mp
import keyboard
import time
import torch
from PIL import Image
import numpy as np
import cv2


class TelloDrone(Tello):
    def __init__(self):
        Tello.__init__(self)
        self.connect()
        print("battery: ",self.get_battery())

        # load model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.lock_class_id = None
        # keyboard control
        self.control_speed = [0,0,0,0] 
        self.keyboard_thread = Thread(target=self.getKeyboardInput)
        self.keyboard_thread.start()
        
        # pose estimation
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        
        # drone video
        self.streamon() 
        self.cap = self.get_frame_read()
        self.drone_frame()
        

    def getKeyboardInput(self):

        while True:
            self.control_speed = [0,0,0,0]
            speed = 50
            # 左右
            if keyboard.is_pressed("LEFT"): self.control_speed[0]=-speed
            elif keyboard.is_pressed("RIGHT"): self.control_speed[0]= speed
            
            # 前後
            if keyboard.is_pressed("UP"): self.control_speed[1]= speed
            elif keyboard.is_pressed("DOWN"): self.control_speed[1]=-speed
            
            # 上下
            if keyboard.is_pressed("w"): self.control_speed[2]= speed
            elif keyboard.is_pressed("s"): self.control_speed[2]=-speed
            
            # 旋轉
            if keyboard.is_pressed("a"): self.control_speed[3]=-speed
            elif keyboard.is_pressed("d"): self.control_speed[3]= speed
            
            # 降落
            if keyboard.is_pressed("q"): self.land(); time.sleep(3) 
            
            # 起飛
            if keyboard.is_pressed("e"): self.takeoff()
            
            # flip 
            if keyboard.is_pressed("j"): self.flip_left(); time.sleep(1)
            elif keyboard.is_pressed("l"): self.flip_right(); time.sleep(1)
            elif keyboard.is_pressed("i"): self.flip_forward(); time.sleep(1)
            elif keyboard.is_pressed("k"): self.flip_back(); time.sleep(1)
            
            time.sleep(0.05) 
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
            # assumes that only each class will only have one object
            for det in self.detections:
                x_center, y_center, width, height, confidence, class_id = det
                if abs(x-x_center) < width/2 and abs(y-y_center) < height/2:
                    self.lock_class_id = class_id

        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Right button clicked at ({x}, {y})")

    def drone_frame(self):
        pTime = 0
        while True:
            image = self.cap.frame
            results = self.model(image)
            self.detections = results.xywh[0].cpu().numpy()
            for det in self.detections:
                x_center, y_center, width, height, confidence, class_id = det
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                label = f"{results.names[int(class_id)]} {confidence:.2f}"
                if(class_id == self.lock_class_id):
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Detected: {results.names[int(class_id)]} with confidence {confidence:.2f}")
                print(f"Bounding Box: (x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height})")

            cv2.imshow('Detection Result', image)
            cv2.setMouseCallback('Detection Result', self.mouse_callback, param=None) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.streamoff()
        cv2.destroyAllWindows()
        self.end()
        
if __name__ == '__main__':
    TelloDrone()