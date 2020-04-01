# *******************************************************************
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/


import argparse
import sys
import os

from utils import *
import datetime
import pandas as pd
import numpy as np


import os
import time

import sys
[sys.path.append(i) for i in ['.', '..']]



#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')
args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('[i] Path to image file: ', args.image)
print('[i] Path to video file: ', args.video)
print('###########################################################\n')



# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
output_dir = 'detected_images'

def detect_face(cap, output_file):
    has_frame, frame = cap.read()

    
  
        
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                    [0, 0, 0], 1, crop=False)

    # # Sets the input to the network
    
    net.setInput(blob)

    # # Runs the forward pass to get output of the output layers
    
    out = get_outputs_names(net)
    
    
    
    
    start_time = time.time()
    outs = net.forward(out)
    
    elapsed_time = time.time() - start_time
    print("*"*70)
    print('inference time cost: {}'.format(elapsed_time))
    print("*"*70)

    # # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, output_file)
    
    # elapsed_time = time.time() - start_time
    # print('inference time cost: {}'.format(elapsed_time)) 
    


    # # initialize the set of information we'll displaying on the frame
    info = [
        ('f ', '{}'.format(len(faces)))
    ]

    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv2.putText(frame, text, (10, (i * 20) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        
    return frame


def _main():

    

    f = open("image.txt", "r")
    print(datetime.datetime.now())
    for item in f:
        # print(x)
        item = os.path.join(os.getcwd(), item.strip())
        # print(item, type(item))
        os.path.join(os.getcwd(), item)
        if not os.path.isfile(item):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            pass
        cap = cv2.VideoCapture(item)
        # print( len(item[:-4].rsplit('\\')))
        # # print(item[:-4].rsplit('/')[])
        # print(item[:-4])
        output_file = item[:-4].rsplit('\\')[-1] + '_yoloface.jpg'
        # print(output_file)
        
        frame = detect_face(cap, output_file)
        cv2.imwrite(os.path.join(output_dir, output_file), frame.astype(np.uint8))
        cap.release()
        # break
        
    print(datetime.datetime.now())


if __name__ == '__main__':
    _main()
