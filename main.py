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
from datetime import datetime
import os
import logging
import warnings
class TelloDrone(Tello):
    def __init__(self):
        Tello.__init__(self)
        self.connect()
        print("battery: ",self.get_battery())
        # ignore warngings
        logging.getLogger('djitellopy').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*deprecated.*")
        # load model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lock_class_id = None
        # control
        self.limit = 70
        self.frame_center = (480, 360)
        self.lr = 0
        self.fb = 0
        self.ud = 0
        self.yv = 0
        self.space_state = 0
        self.rev_on = False
        self.rev_speed = 0
        self.keyboard_thread = Thread(target=self.getKeyboardInput)
        self.keyboard_thread.start()
        self.pid_fb = PID(0.105, 0.001, 0.05, setpoint=0)
        self.pid_ud = PID(0.1, 0.001, 0.05, setpoint=0)
        self.pid_yv = PID(0.11, 0.001, 0.05, setpoint=0)
        self.update_width_flag = False
        self.target_width = 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(current_time)
        self.setting_file_path = os.path.join(current_time, "settings.txt")
        self.log_file_path = os.path.join(current_time, "log.txt")
        with open(self.setting_file_path, "w") as log_file:
            log_file.write(str(self.pid_fb) + "\n")
            log_file.write(str(self.pid_ud) + "\n")
            log_file.write(str(self.pid_yv) + "\n")
            log_file.write(str(self.limit) + "\n")

        # pose estimation
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        
        # drone video
        self.streamon() 
        self.cap = self.get_frame_read()
        self.drone_frame()
        
    def getKeyboardInput(self):
        step = 5
        while True:
            self.lr, self.fb, self.ud, self.yv = 0,0,0,0
            speed = 50
            if keyboard.is_pressed("LEFT"): self.lr = -speed
            elif keyboard.is_pressed("RIGHT"): self.lr = speed

            if keyboard.is_pressed("UP"): 
                self.fb = speed
                self.update_width_flag = True
                
            elif keyboard.is_pressed("DOWN"): 
                self.fb = -speed
                self.update_width_flag = True
            
            if keyboard.is_pressed("w"): self.ud = speed
            elif keyboard.is_pressed("s"): self.ud = -speed
 
            if keyboard.is_pressed("a"): self.yv = -speed
            elif keyboard.is_pressed("d"): self.yv = speed
            
            if keyboard.is_pressed("q"): self.land(); time.sleep(3) 
            
            if keyboard.is_pressed("e"): self.takeoff()

            if keyboard.is_pressed("1") and self.rev_speed > -100:
                self.rev_speed-=step
                print("revolution speed: ", self.rev_speed)
            
            if keyboard.is_pressed("2") and self.rev_speed < 100:
                self.rev_speed+=step
                print("revolution speed: ", self.rev_speed)
            
            if keyboard.is_pressed("space") and self.space_state == 0:
                self.space_state = 1
            if not keyboard.is_pressed("space") and self.space_state == 1:
                self.space_state = 0
                self.rev_on = not self.rev_on
                print("revolution on: ",self.rev_on)

            # # flip 
            # if keyboard.is_pressed("j"): self.flip_left(); time.sleep(1)
            # elif keyboard.is_pressed("l"): self.flip_right(); time.sleep(1)
            # elif keyboard.is_pressed("i"): self.flip_forward(); time.sleep(1)
            # elif keyboard.is_pressed("k"): self.flip_back(); time.sleep(1)
            
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
                    self.target_width = width

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.lock_class_id = None
            print(f"Right button clicked at ({x}, {y})")

    def adj_pose(self, ud_dif,fb_dif,yv_dif):
        if(self.rev_on): self.lr = self.rev_speed
        else: self.lr = 0
        self.fb = int(self.pid_fb(fb_dif))
        self.ud = int(self.pid_ud(ud_dif))
        self.yv = -int(self.pid_yv(yv_dif))
        
        for val in [self.lr, self.fb, self.ud, self.yv]:
            if val > self.limit: val = self.limit
            elif val < -self.limit: val = -self.limit

    def drone_frame(self):
        pTime = 0
        while True:
            pTime = round(time.time(), 3)
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
                    if(self.update_width_flag):
                        self.update_width_flag = False
                        self.target_width = width
                    ud_dif = y_center - self.frame_center[1]
                    fb_dif = width - self.target_width
                    yv_dif = x_center - self.frame_center[0]
                    self.adj_pose(ud_dif, fb_dif, yv_dif)
                    if(self.fb>0):
                        cv2.circle(image, self.frame_center, self.fb, (0, 255, 255), 2)
                    else:
                        cv2.circle(image, self.frame_center, self.fb*-1, (0, 0, 255), 2)
                    cv2.arrowedLine(image, self.frame_center, (self.frame_center[0],self.frame_center[1]-self.ud), (0, 255, 255), 2)
                    cv2.arrowedLine(image, self.frame_center, (self.frame_center[0]+self.yv,self.frame_center[1]), (0, 255, 255), 2)

                    cv2.circle(image, (int(x_center), int(y_center)), 2, (255, 0, 0), 2)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    bounding_box = [
                        round(float(x_center), 3),
                        round(float(y_center), 3),
                        round(float(width), 3),
                        round(float(height), 3)
                    ]
                    differences = [
                        round(float(ud_dif), 3),
                        round(float(fb_dif), 3),
                        round(float(yv_dif), 3)
                    ]
                    with open(self.log_file_path, "a") as log_file:
                        log_message = (
                            f"Time: {pTime};"
                            f"bat: {str(self.get_battery())};"
                            f"{str(bounding_box)};"
                            f"{str(differences)};"
                            f"{str([self.lr, self.fb, self.ud, self.yv])}\n"
                        )
                        log_file.write(log_message)
                else:
                    # self.lr, self.fb, self.ud, self.yv = 0,0,0,0
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, self.frame_center, 2, (255, 255, 255), 2)
            cv2.imshow('Detection Result', image)

            self.send_rc_control(self.lr, self.fb, self.ud, self.yv)
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