
import numpy as np
import glob
import cv2
import os


class DataGenerator:
    def __init__(self, path, shape, n_frames = 5, stride=6):
        self.stable_path = os.path.join(path,'stable')
        self.unstable_path = os.path.join(path,'unstable')
        self.length = n_frames
        self.stride = stride
        self.shape = shape
        self.video_names = os.listdir(self.stable_path)
        self.frame_idx = 30

    def get_paths(self,video):
        s_path = os.path.join(self.stable_path,video)
        u_path = os.path.join(self.unstable_path,video)
        paths = [s_path,u_path]
        return( paths)
    
    def __call__(self):
        for video in self.video_names:
            paths = self.get_paths(video)
            stable_frames, unstable_frames = load_video(paths,self.shape)
            n,h,w,c = stable_frames.shape
            sequence = np.zeros(shape=(h,w,self.length),dtype=np.float32)
            It = np.zeros(shape=(h,w,c),dtype=np.float32)
            Igt = np.zeros_like(It)
            for frame_idx in range(30,n):
                for (i,j) in zip(range(frame_idx - self.stride, frame_idx - self.length*self.stride, -self.stride) , range(self.length)):
                    sequence[:,:,j] = cv2.cvtColor(stable_frames[i,...],cv2.COLOR_BGR2GRAY)
                It = unstable_frames[frame_idx,...]
                Igt = stable_frames[frame_idx,...]
                yield sequence, It, Igt


def load_video(paths,shape):
    stable_frames = []
    unstable_frames = []
    stable_cap = cv2.VideoCapture(paths[0])
    unstable_cap = cv2.VideoCapture(paths[1])
    while True:
        ret, frame1 = stable_cap.read()
        if not ret:
            break
        frame1 = preprocess(frame1,shape)
        stable_frames.append(frame1)
        ret, frame2 = unstable_cap.read()
        if not ret:
            break
        frame2 = preprocess(frame2,shape)
        unstable_frames.append(frame2)
    stable_cap.release()
    unstable_cap.release()
    #in some video pairs the stable and unstable version dont have the same frame count
    frame_count = min(len(stable_frames),len(unstable_frames))
    stable_frames = stable_frames[:frame_count]
    unstable_frames = unstable_frames[:frame_count]
    #convert to np.arrays
    stable_frames = np.array(stable_frames,dtype=np.float32)
    unstable_frames = np.array(unstable_frames,dtype=np.float32)
    return(stable_frames,unstable_frames)

def preprocess(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = img / img.max()
    return img

def get_samples(stable,unstable,length,stride):
    '''returns all samples from a video
    '''
    n,h,w,c = stable.shape
    sequence = np.zeros(shape=(n-29,h,w,length), dtype=np.float32)
    It = np.zeros(shape=(n-29,h,w,c),dtype=np.float32)
    Igt = np.zeros_like(It)
    for frame_idx in range(30,n):
        for (i,j) in zip(range(frame_idx - stride, frame_idx - length*stride, -stride) , range(length)):
            sequence[frame_idx-30,:,:,j] = cv2.cvtColor(stable[i,...],cv2.COLOR_BGR2GRAY)
        It[frame_idx-30,...] = unstable[frame_idx,...]
        Igt[frame_idx-30,...] = stable[frame_idx,...]
    return sequence, It, Igt


