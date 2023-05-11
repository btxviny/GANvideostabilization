import cv2
import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import os 

def make_RGB(img):
    img = np.expand_dims(img,-1)
    img = np.tile(img, [1,1,3])
    return(img)

class DataLoader(Sequence):
    def __init__(self, path, length, target_shape, batch_size, steps_per_epoch):
        self.path = path
        self.length = length
        self.batch_size = batch_size#213
        self.steps_per_epoch = steps_per_epoch
        self.shape = target_shape
        #dataset attributes
        self.stable_path = os.path.join(self.path,'stable')
        self.unstable_path = os.path.join(self.path,'unstable')
        self.videos = os.listdir(os.path.join(self.path,'stable')) #list of video names
        #################################################################
        self.video_idx = 0
        self.frame_idx = 30 
        self.total_frames = 0
        self.stable_cap = None
        self.unstable_cap = None
        ################################################################
        self.stable_frames = None
        self.unstable_frames = None
        self.A = np.zeros(shape = (self.length,)+self.shape, dtype=np.float32)
        self.A_tensor =  tf.Variable(initial_value = tf.zeros(shape = (self.batch_size,)+self.shape[:-1]+(15,)), dtype = tf.float32)
        #################################################################
    
    def __len__(self):
        return(self.steps_per_epoch)
    
    def on_epoch_end(self):
        return 
        
    def get_item(self): 
        # Get the current batch of frames
        unsteady = np.zeros(shape=(self.batch_size,)+ self.shape, dtype=np.float32)
        ground_truth = np.zeros_like(unsteady)
        sequence = np.zeros(shape=(self.batch_size,) + self.shape[:-1] + (self.length,), dtype=np.float32)
        for idx in range(self.batch_size):
            sequence[idx,...], unsteady[idx,...], ground_truth[idx,...] = self.__get_data()
            
        return [sequence, unsteady, ground_truth] 



    def __getitem__(self, idx): 
        # Get the current batch of frames
        unsteady = np.zeros(shape=(self.batch_size,)+ self.shape, dtype=np.float32)
        ground_truth = np.zeros_like(unsteady)
        sequence = np.zeros(shape=(self.batch_size,) + self.shape[:-1] + (self.length,), dtype=np.float32)
        for idx in range(self.batch_size):
            sequence[idx,...], unsteady[idx,...], ground_truth[idx,...] = self.__get_data()   
        return [sequence, unsteady, ground_truth]


    def __get_data(self):
        if self.video_idx == 0:
            self.__load_video()
        elif self.video_idx == len(self.videos) -1 :
            self.video_idx = 0
            self.__load_video()
        if self.frame_idx >= self.total_frames:
            self.video_idx += 1
            self.frame_idx = 30
            self.__load_video()
        It = self.unstable_frames[self.frame_idx,...]
        Igt = self.stable_frames[self.frame_idx,...]

        step = 30 // (self.length -1)
        St = np.zeros(shape = self.shape[:-1]+(self.length,), dtype=np.float32)
        for (i ,j) in zip(range(self.frame_idx - step, self.frame_idx - (30+step), -step) , range(self.length)):# negative step so the most recent frame goes first in the least
            grayscale = cv2.cvtColor(self.stable_frames[i], cv2.COLOR_BGR2GRAY)
            St[:,:,j] = grayscale                                         
        
        self.frame_idx += 1
        return St, It, Igt

    def __load_video(self):
        self.stable_frames = []
        self.unstable_frames = []
        h,w,_ = self.shape
        stable_cap = cv2.VideoCapture(os.path.join(self.stable_path,self.videos[self.video_idx]))
        unstable_cap = cv2.VideoCapture(os.path.join(self.unstable_path,self.videos[self.video_idx]))
        while True:
            ret, frame1 = stable_cap.read()
            if not ret:
                break
            frame1 = frame1 / frame1.max()
            frame1 = cv2.resize(frame1,(w,h))
            self.stable_frames.append(frame1)
            ret, frame2 = unstable_cap.read()
            if not ret:
                break
            frame2 = frame2 / frame2.max()
            frame2 = cv2.resize(frame2,(w,h))
            self.unstable_frames.append(frame2)
        #frame arrays
        self.total_frames =min(len(self.stable_frames),len(self.unstable_frames))
        self.unstable_frames = self.unstable_frames[:self.total_frames]
        self.stable_frames = self.stable_frames[:self.total_frames]
        self.stable_frames , self.unstable_frames = np.array(self.stable_frames, dtype=np.float32) , np.array(self.unstable_frames, dtype=np.float32)
    def get_A(self):
        self.__load_video()
        for (i,j) in zip(range(self.frame_idx-1, self.frame_idx - self.length -1, -1), range(self.length)):
            self.A[j,:,:,:] = self.stable_frames[i]
        temp = self.A.copy().reshape(256,256,15)
        temp= tf.convert_to_tensor(temp)
        temp = tf.expand_dims(temp, axis = 0 )
        self.A_tensor.assign(temp)
        return self.A_tensor
    
