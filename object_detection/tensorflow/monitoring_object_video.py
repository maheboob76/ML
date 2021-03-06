
''' This will monitor teddy continously and alert if teddy is moved/stolen '''
import cv2
import numpy as np
from datetime import datetime
from playsound import playsound

import threading
from _thread import start_new_thread
from time import sleep
# Load a model imported from Tensorflow
''' MOBILE NET net returns 0 based class ids wheeras all other are returning 1 based'''
#MODEL_PATH = 'models/MobileNet-SSD v2/ssd_mobilenet_v2_coco_2018_03_29/'
MODEL_PATH = 'pre_trained_models/Faster-RCNN Inception v2/faster_rcnn_inception_v2_coco_2018_01_28/'


CONF_THRESHOLD = .3
COCO_LABELS = 'coco-labels.txt'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

CV_WINDOW_HEIGHT = 1200
CV_WINDOW_WIDTH = 1600
CUSTOM_CLASSES = [56,88]
TOTAL_CLASSES = 91
    
#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(TOTAL_CLASSES, 3))
IS_OBJ_AVAILABLE = True
ALERT_FLAG = True
AUDIO_FILE = 'audios/Siren.mp3'

CLASSES = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


#waitkey_demo()


def waitkey_demo():
    img = cv2.imread('ToyGun1.jpg') # load a dummy image
    while(1):
        cv2.imshow('img',img)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print( k) # else print its value


tensorflowNet = cv2.dnn.readNetFromTensorflow(MODEL_PATH + 'frozen_inference_graph.pb', MODEL_PATH + 'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)

cv2.namedWindow('amaan', cv2.WINDOW_NORMAL)
cv2.resizeWindow('amaan',CV_WINDOW_WIDTH, CV_WINDOW_HEIGHT)


def generate_alert():
    global IS_OBJ_AVAILABLE
    global ALERT_FLAG
    
    while ALERT_FLAG:
        thread_name = threading.get_ident() 
        current_time = datetime.now()
        print('{} Thread: {} Checking alert '.format(current_time, thread_name))
        
        if  not IS_OBJ_AVAILABLE:
            playsound(AUDIO_FILE)
            #IS_OBJ_AVAILABLE = False
            
        sleep(1)
        
    #playsound('audios/Tornado_Siren.mp3')
    

# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, left, top, right, bottom, p, thickness=2):
    
    global IS_OBJ_AVAILABLE
	
    # This is added as background may not be available in all the modesl
    class_id +=1
    
    if class_id not in CUSTOM_CLASSES:
       #count =+1
       IS_OBJ_AVAILABLE = False
       return 

    #threaded invocation

    IS_OBJ_AVAILABLE = True
    
    
    #generate_alert( datetime.now(), CLASSES.get(class_id))
    #label = str(classes[class_id])

    #color = COLORS[class_id]
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)
    class_name = CLASSES.get(class_id)
    label = class_name + '{0:.2f}'.format(confidence)
    color = COLORS[class_id]
    
    #cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
    
    cv2.putText(img, label, (left + 10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 2)
    


# check and alert in its own thread


# Define video capture for default cam    
cap = cv2.VideoCapture(0)


def startMoniotirng():
    monitor_thread = start_new_thread(generate_alert,())    
    global ALERT_FLAG
    global IS_OBJ_AVAILABLE
    
    while(1):
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            ALERT_FLAG = False
            break

        hasframe, image = cap.read()
        
        if image is None:
            print('No image found.....ALERT for it')
            IS_OBJ_AVAILABLE = False
            continue
        
        rows, cols, channels = image.shape
        #image=cv2.resize(image, (620, 480)) 
        
        blob = cv2.dnn.blobFromImage(image, size=(IMAGE_WIDTH, IMAGE_WIDTH), swapRB=True, crop=False)
        tensorflowNet.setInput(blob)
        
        
        # Runs a forward pass to compute the net output
        networkOutput = tensorflowNet.forward()
        obj_count = 0
        # Loop on the outputs
        for detection in networkOutput[0,0]:
            
            confidence = float(detection[2])
            
            #scores = detection[5:]#classes scores starts from index 5
             
            if confidence > CONF_THRESHOLD:
            	
                obj_count += 1
                
                class_id = int(detection[1])
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
               
                #draw a red rectangle around detected objects
                #cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                draw_pred(image, class_id, confidence, left, top, right, bottom, (0, 0, 255), thickness=2)
                
       
         # Put efficiency information.
        t, _ = tensorflowNet.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        
        # Show the image with a rectagle surrounding the detected objects 
        cv2.imshow('amaan', image)
    

if __name__ == '__main__':
    startMoniotirng()