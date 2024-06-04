#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    print("D3QN sys path:", sys.path)

import cv2
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

#from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import Conv2D
from keras.layers import Input, Dense, Flatten, Lambda, add
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential ,load_model, Model
#from keras.backend.tensorflow_backend import set_session
#Does not work from tensorflow.keras.backend import set_session
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Vector3Stamped, Twist
from skimage.transform import resize
from PIL import Image as iimage
from keras.models import load_model
#from keras.utils.training_utils import multi_gpu_model
from ENV import environment
from drone_control import PID
from std_msgs.msg import Empty
from matplotlib import gridspec
from matplotlib.figure import Figure
import Depth_Anything.depth_anything
from Depth_Anything.run import run_DA
import action_evaluation
import zone_characterisation

import matplotlib.pyplot as plt
import rospy
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import scipy.misc
import numpy as np
import random
import time
import random
import pickle
import models
import cv2
import copy
#New
import subprocess
from cv_bridge import CvBridge
import os


image = []
dep_image=[]

def callback_camera(msg):
    global image
    img_height = msg.height
    img_width = msg.width
        
    image = np.frombuffer(msg.data, dtype=np.uint8)       
    image = np.reshape(image, [img_height,img_width,3]) 
    image = np.array(image)
"""
def callbackcam_depth(msg):
    global dep_image
    bridge = CvBridge()
    dep_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") 

    cv2.imshow("Image", dep_image)
    cv2.waitKey(1)
"""
def speed_plot(a,b):
    linear_speed = a
    angular_speed = b
    
    angular_list = (['angular_velocity'])
    linear_list = (['linear velocity'])
    
    x_pos = np.arange(len(angular_list))
    y_pos = np.arange(len(linear_list))
    
    fig = plt.figure(1)
    #plot figure with subplots of different sizes
    gridspec.GridSpec(6,6)
    #set up subplot grid    
    plt.subplot2grid((6,6), (1,0), colspan=4, rowspan=2)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.title('Angular (rad/s)')
    plt.xlabel('value')
    plt.barh(y_pos, [-0.6, 0.6], color = [1.0,1.0, 1.0])
    plt.barh(y_pos, angular_speed, color = 'r')
    plt.yticks([])
    
    plt.subplot2grid((6,6), (0,5), colspan=1, rowspan= 5)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.locator_params(axis = 'y', nbins = 5)
    plt.title('Linear (m/s)')
    plt.xlabel('value')
    plt.bar(x_pos, [0, 0.2], color = [1.0, 1.0, 1.0])
    plt.bar(x_pos, linear_speed, color = 'b')
    plt.xticks([])
    fig.savefig('sample.png', dpi=fig.dpi, bbox_inches = 'tight')

    
def realtime_plot_cv2(a,b):
    speed_plot(a,b)
    img = cv2.imread('sample.png', cv2.IMREAD_COLOR)
    cv2.imshow('control_pannel',img)
    cv2.waitKey(1)


# load model
g1 = tf.Graph()
g2 = tf.Graph()


with g1.as_default():
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    #model_data_path = './NYU_FCRN-checkpoint/NYU_FCRN.ckpt'

    # Construct the network
    print('start create the session and model')
    #net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    #cnn_init = tf.global_variables_initializer()
    #cnn_config = tf.ConfigProto(allow_soft_placement = True)
    #cnn_sess = tf.Session(config = cnn_config)
    #cnn_sess.run(cnn_init)
    #cnn_saver = tf.train.Saver()     
    #cnn_saver.restore(cnn_sess, model_data_path)

    print('Finishied')
    
def get_depth(image):
    raw_img=np.array(image)
    depth_image = run_DA.main_DA(raw_img)

    #Construct the command to run run.py
    #python3 run.py --encoder vits --img-path /home/jaume/catkin_ws/src/Collision-avoidance/Images/FPD.jpg --outdir /home/jaume/catkin_ws/src/Output_images --pred-only --grayscale
    #print("Running Depth Anything")
    #command=["python3", "run.py", "vits", "--img-path", "/home/jaume/catkin_ws/src/Collision-avoidance/Images/Foto_prova.jpg", "--outdir", "/home/jaume/catkin_ws/src/Output_images", "--pred-only", "grayscale"]
    #subprocess.run(command)
    #depth_image=iimage.open('/home/jaume/catkin_ws/src/Collision-avoidance/Output_images/Foto_prova_depth.jpg')
    #depth_image=np.array(depth_image)
    depth_image=iimage.fromarray(depth_image)

    pred=depth_image.resize([160, 128], iimage.LANCZOS)
    pred=pred.convert('L')
    pred=np.array(pred)
    pred=255-pred
    #pred = np.reshape(pred, [128,160])   
    #pred= np.array(pred, dtype=np.float32)
    
    #pred[np.isnan(pred)] = 5. 
    #pred = pred / 3.5
    #pred[pred>1.0] = 1.0
    
    return pred

class TestAgent:
    def __init__(self, action_size):
        self.state_size = (128, 160 , 8)
        self.action_size = action_size
        self.model = self.build_model()
        self.config = tf.ConfigProto()                
        self.sess = tf.InteractiveSession(config=self.config)         
        self.sess.run(tf.global_variables_initializer())
        #K.set_session(self.sess)           
    
    def build_model(self):
        input = Input(shape=self.state_size)
        h1 = Conv2D(32, (10, 14), strides = 8, activation = "relu", name = "conv1")(input)
        h2 = Conv2D(64, (4, 4), strides = 2, activation = "relu", name = "conv2")(h1)
        h3 = Conv2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3")(h2)
        context = Flatten(name = "flatten")(h3)
        
        value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(context)
        value = Dense(1, name = "value")(value_hidden)
        action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(context)
        action = Dense(self.action_size, name = "action")(action_hidden)
        action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keepdims = True), name = 'action_mean')(action) 
        output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'output')([action, value, action_mean]) #
        model = Model(inputs = input, outputs = output)
        model.summary()
        
        return model

    def get_action(self, history):
        flag = False
        if np.random.random() < 0.001:
            flag = True
            return random.randrange(8), flag 
        history = np.float32(history)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0]), flag

    def load_model(self, filename):
        self.model.load_weights(filename)        
        
# Intial hovering for 2 seconds and collecting the data from the laser and camera
# Receivng the data from laser and camera
# checking the crash using crash_check function and if crash occurs, the simulation is reset
with g2.as_default():
    
    if __name__ == '__main__':
        index_action=0
        "Subscribe"
        rospy.init_node('D3QN_TEST', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, callback_camera, queue_size = 10)
        #The name of the ros topic from Xinghua /camera/infra1/image_rect_raw
        #RGB /camera/color/image_raw  -- grayscale /camera/depth/image_rect_raw
        # Parameter setting for the simulation
        agent = TestAgent(action_size = 8)  ## class name should be different from the original one
        #Change agent.load_model("./Saved_models/D3QN_V_3_single.h5")
        agent.load_model("/home/jaume/catkin_ws/src/Collision-avoidance/real_world_test/Saved_models/D3QN_V_17_single.h5")
        EPISODE = 100000
        global_step = 0
       
        env = environment()
        # Observe    
        rospy.sleep(2.)                
        e = 0
        rate = rospy.Rate(5)
        #vel_pid = PID()
        
        while e < EPISODE and not rospy.is_shutdown():
            rospy.sleep(1)
            
            e = e + 1   
            # get the initial state
            state = get_depth(image)
            
            #Image processing: floor removal
            #n_rows=depth_image.shape[0]
            #n_columns=depth_image.shape[1]
            n_rows=state[:,1].size
            n_columns=state[1,:].size
            #print('Number of rows: ', n_rows)
            #print('Number of columns: ', n_columns)
            bright_pix=np.argmin(state)
            row_index=bright_pix//n_columns
            column_index=bright_pix%n_columns
            pix_brightness=state[row_index, column_index]
            
            """
            if row_index>120:
                state=state[0:120,:]
                state=iimage.fromarray(state)
                state=state.resize([160, 128], iimage.LANCZOS)
                state=state.convert('L')
                state=np.array(state)
            """
            bright_pix=np.argmin(state)
            row_index=bright_pix//n_columns
            column_index=bright_pix%n_columns
            pix_brightness=state[row_index, column_index]
            
            history = np.stack((state, state, state, state, state, state, state, state), axis = 2)                
            history = np.reshape([history], (1,128,160,8))        

            step, score  = 0. ,0.
            done = False                    

            while not done and not rospy.is_shutdown():  
                
                #global_starting_time=rospy.Time.now() 
                
                global_step = global_step + 1             
                step = step + 1   
                # Receive the action command from the Q-network and do the action           
                [action, flag] = agent.get_action(history)
                #print('We are in step: ', step)
                #Action Policy
                next_state=[]
                previous_action=[]
                #Reinitialize brightest pixel coordenates
                #Danger Zone <1.5m
                """
                print('Initial action was:', action)
                if pix_brightness<160:
                    print('Danger Zone')
                    if action==1 or action==0 or action==5:
                        action=2
                    
                #Safe Zone <2m, Prevent unnecessary stops
                elif pix_brightness>160 and pix_brightness<180:
                    print('Safe Zone')
                    if previous_action==2 and action==2:
                        action=1 
                
                #Confort zone <4m, no heavy turns
                else:
                    print('Confort Zone')
                    if action==3:
                        action=7 
                    elif action==4:
                        action=6 
                """      
                print("The action would be: ", action)
                print("Step number:", step)
                
                pixel=(row_index,column_index)
                zone=zone_characterisation.zone_det(pixel)
                val=action_evaluation.f_action(zone, pix_brightness, action) #zone to be initialized
                index_action=index_action + val
                print('Index:', index_action)
                print('Zone: ', zone)
                print("------------------------------")
                print("------------------------------")
                print("------------------------------")
                              
                image_number=step-1
                #print('Corresponding Image Number', image_number)
                image_name=f'image_leftapproach_{image_number}.jpg'
                save_dir='/home/jaume/jaume_ws/src/Collision-avoidance/real_world_test/Depth_Anything/Output_Images'
                #cv2.imwrite(os.path.join(save_dir,image_name), image)                      
                
                #give control_input 
                #[linear, angular] = vel_pid.velocity_control(action)
    
                # image preprocessing           
                starting_time=rospy.Time.now() 
                next_state = get_depth(image)
                ending_time=rospy.Time.now()
                processing_time=ending_time-starting_time
                processing_time=processing_time.to_sec()
                #print('Processing time of Depth-Anything: ', processing_time)
                
                n_rows=next_state[:,1].size
                n_columns=next_state[1,:].size
                #print('Number of rows: ', n_rows)
                #print('Number of columns: ', n_columns)
                bright_pix=np.argmin(next_state)
                row_index=bright_pix//n_columns
                column_index=bright_pix%n_columns
                """
                if row_index>120:
                    print('Image cut and reshaped')
                    next_state=next_state[0:120,:]
                    next_state=iimage.fromarray(next_state)
                    next_state=next_state.resize([160, 128], iimage.LANCZOS)
                    next_state=next_state.convert('L')
                    next_state=np.array(next_state)
                 """
                n_rows=next_state[:,1].size
                n_columns=next_state[1,:].size
                bright_pix=np.argmin(next_state)
                row_index=bright_pix//n_columns
                column_index=bright_pix%n_columns
                pix_brightness=next_state[row_index, column_index]
                print('Brightest pixel is:', [row_index, column_index])
                print('With a brightness of: ', pix_brightness)
                #print('Closest pixel birghtness:', next_state[row_index,column_index])
                           
                #Save depth image
                #depth_image_name=f'depth_image_leftapproach_{image_number}.jpg'
                #cv2.imwrite(os.path.join(save_dir,depth_image_name), next_state)

                
                #Left chair - [40,30] || central chair - [40,75] || Right chair [20, 145]
                #print('Left chair brightness: ', next_state[30,30])
                #print('Central chair brightness: ', next_state[30,85])
                #print('Right chair brightness: ', next_state[30,135])

                # plot real time depth image
                aa = cv2.resize(next_state, (160*3, 128*3), interpolation = cv2.INTER_CUBIC)
                plt.imshow(image)
                #plt.show()
                cv2.imshow('input image', aa)
                cv2.waitKey(1)
                
                #plot real time control input
                #realtime_plot_cv2(linear, -angular)
                
                # image for collision check            
                next_state = np.reshape([next_state],(1,128,160,1))                        
                next_history = np.append(next_state, history[:,:,:,:7],axis = 3)
                next_history = np.append(next_state, history[:,:,:,:7],axis = 3)
                next_history = np.append(next_state, history[:,:,:,:7],axis = 3)
                next_history = np.append(next_state, history[:,:,:,:7],axis = 3)
                
                history = next_history
                previous_action=action

                #global_ending_time=rospy.Time.now()
                #global_processing_time=global_ending_time-global_starting_time
                #global_processing_time=global_processing_time.to_sec()
                #print('Processing time of iteration: ', global_processing_time)

                if step >= 2000:
                    done = True
                # Update the score            
                rate.sleep()

            if done:    
                print(score)
                
