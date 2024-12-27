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
from simple_pid import PID

class TelloDrone(Tello):
    def __init__(self):
        Tello.__init__(self)
        self.connect()
        print("battery: ",self.get_battery())

        # load model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.lock_class_id = None
        # control
        self.frame_x_center = 480
        self.frame_y_center = 360
        self.control_speed = [0,0,0,0] 
        self.keyboard_thread = Thread(target=self.getKeyboardInput)
        self.keyboard_thread.start()
        self.pid_fb = PID(0.3, 0.001, 0.05, setpoint=0)
        self.pid_ud = PID(0.3, 0.001, 0.05, setpoint=0)
        self.pid_yv = PID(0.3, 0.001, 0.05, setpoint=0)
        self.target_width = 200

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
    
    # "self" was added to the argument to obtain class attribute
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
    
    def adj_pose(self, x_center, y_center, width, height):
        # self.control_speed[0]
        ud_dif = y_center - self.frame_y_center
        fb_dif = width - self.target_width
        yv_dif = x_center - self.frame_x_center
        self.control_speed[1] = int(self.pid_ud(ud_dif))
        self.control_speed[2] = int(self.pid_fb(fb_dif))
        self.control_speed[3] = int(self.pid_yv(yv_dif))
        print(ud_dif,fb_dif,yv_dif)
        print(self.control_speed)

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
                    self.adj_pose(x_center, y_center, width, height)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    self.control_speed = [0,0,0,0]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # print(f"Detected: {results.names[int(class_id)]} with confidence {confidence:.2f}")
                # print(f"Bounding Box: (x_center: {x_center}, y_center: {y_center}, width: {width}, height: {height})")

            cv2.imshow('Detection Result', image)
            # A argument "self" was added to mouse_callback function so parameter has to be None
            # otherwise it will be count as a positional argument and cause TypeError
            cv2.setMouseCallback('Detection Result', self.mouse_callback, param=None) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.streamoff()
        cv2.destroyAllWindows()
        self.end()
        
if __name__ == '__main__':
    TelloDrone()