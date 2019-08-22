# How to load a Tensorflow model using OpenCV
# Jean Vitor de Paulo Blog - https://jeanvitor.com/tensorflow-object-detecion-opencv/

''' Able to detect people , bed , chair etc . struggled with toothbrush, books etc '''
import cv2
import numpy as np

# Load a model imported from Tensorflow
''' MOBILE NET net returns 0 based class ids wheeras all other are returning 1 based'''
#MODEL_PATH = 'models/MobileNet-SSD v2/ssd_mobilenet_v2_coco_2018_03_29/'
MODEL_PATH = 'models/Faster-RCNN Inception v2/faster_rcnn_inception_v2_coco_2018_01_28/'


CONF_THRESHOLD = .5
COCO_LABELS = 'coco-labels.txt'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

CV_WINDOW_HEIGHT = 600
CV_WINDOW_WIDTH = 800

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




#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(91, 3))

tensorflowNet = cv2.dnn.readNetFromTensorflow(MODEL_PATH + 'frozen_inference_graph.pb', MODEL_PATH + 'faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

#cv2.namedWindow('image',cv2.WINDOW_NORMAL)

cv2.namedWindow('amaan', cv2.WINDOW_NORMAL)
cv2.resizeWindow('amaan',CV_WINDOW_WIDTH, CV_WINDOW_HEIGHT)


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, left, top, right, bottom, p, thickness=2):
    
	
    # This is added as background may not be available in all the modesl
    class_id +=1
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
    

# Define video capture for default cam
cap = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:
    
    hasframe, image = cap.read()
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
