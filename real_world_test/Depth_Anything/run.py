#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    #print("run sys path:", sys.path)

import argparse
import cv2
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

#from depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.dpt import DepthAnything
#from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

#ros adaptation
from sensor_msgs.msg import Image
from std_msgs.msg import String
import rospy
import matplotlib.pyplot as plt
from cv_bridge import CvBridge

"""
run command: python3 run.py --encoder vits --img-path /home/jaume/catkin_ws/src/Collision-avoidance/Depth-Anything/Test_Images/FP1.jpg --outdir /home/jaume/catkin_ws/src/Collision-avoidance/Depth-Anything/Output_Images/ --pred-only --grayscale
python3 run.py --encoder vits --pred-only --grayscale
"""
cv_image=[]
#Adaptation to ros environment
class run_DA():
    #def __init__(self) :
        
    def callback_camera(msg):
        global cv_image
        """
        #Original callback image from D3QN test
        img_height = msg.height
        img_width = msg.width
        
        cv_image = np.frombuffer(msg.data, dtype=np.uint8)       
        cv_image = np.reshape(cv_image, [img_height,img_width,3]) 
         cv_image = np.array(cv_image)
         """
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)

    def main_DA(np_img):
        parser = argparse.ArgumentParser()
     #No need to call the directories if using Subscriber/Publisher functions
    #parser.add_argument('--img-path', type=str)
    #parser.add_argument('--outdir', type=str, default='./vis_depth')
        parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    
        parser.add_argument('--pred-only', dest='pred_only', default='pred_only', action='store_true', help='only display the prediction')
        parser.add_argument('--grayscale', dest='grayscale', default='grayscale', action='store_true', help='do not apply colorful palette')
    
        args = parser.parse_args()
    
        #margin_width = 50
        #caption_height = 60
    
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #font_scale = 1
        #font_thickness = 2
    
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()
    
        #total_params = sum(param.numel() for param in depth_anything.parameters())
        #print('Total parameters: {:.2f}M'.format(total_params / 1e6))
    
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        """
        if os.path.isfile(args.img_path): #If the input directory leads to a given file:
        if args.img_path.endswith('txt'): #If the given file contains more than one image:
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines() #List the different images as different filenames/directories
        else:
            filenames = [args.img_path] #If the given file contains just one image save the file/directory of such image
        else: #If the input directory does not lead to a file (leads to a directory)
        filenames = os.listdir(args.img_path) #list the filenames within the directory
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')] #Construct the absolute path of images that don't start with a . (wouldbe hidden images)
        filenames.sort() #This lines ensures images will be processed in consistent order
    
        os.makedirs(args.outdir, exist_ok=True) #Create the directory specified in args.outdir if it does not exist already
        """
    #for filename in tqdm(filenames): CARE, IF USING THIS LINE, TAB EVERY LINE BELOW
    
    #ROS adaptation
    #Subscriber (Subscribe to the raw_image topic)
    
        #rospy.init_node('camera_subscriber', anonymous=True)
        #image object should be initialized with callback function
        #rospy.Subscriber('/bebop/raw_image', Image, callback_camera, queue_size = 10)
        #print("Instant before spin")
        #rospy.spin()
        #print("Raw Image opened in run file using Ros Subscribers", cv_image)

        image = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB) / 255.0
    
        h, w = image.shape[:2]
      
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
     
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
      
        depth = depth.cpu().numpy().astype(np.uint8)
        
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        return depth
    #filename = os.path.basename(filename)
    """
        #if args.pred_only:
        print('Option 1')
        plt.imshow(depth)
        plt.show()
        #cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        #cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.jpg'), depth)
        #ROS adaptation
        #Publish (publishing the depth_image)
        dpd = rospy.Publisher('/bebop/depth_image', Image, callback_camera , queue_size = 10)
        rospy.init_node('talker', anonymous=True)
        
        depth_data=Image()
        depth_data.header.stamp = rospy.Time.now()
        depth_data.height = depth.shape[0]
        depth_data.width = depth.shape[1]
        depth_data.encoding = 'mono8' #grayscale
        depth_data.is_bigendian = False
        depth_data.step = depth.shape[1]
        depth_data.data = depth.tobytes
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            dpd.publish(depth_data)
            print("After publisher")
            rate.sleep()

        else:
            print('Option 2')
            #split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            split_region = np.ones((np_img.shape[0], margin_width, 3), dtype=np.uint8) * 255
            #combined_results = cv2.hconcat([raw_image, split_region, depth])
            combined_results = cv2.hconcat([np_img, split_region, depth])
           
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            #cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
            #cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.jpg'), final_result)
            #Publish (publishing the depth_image)
            #final_result = rospy.Publisher('/bebop/depth_image', Image, queue_size = 10)
            return final_result
            """