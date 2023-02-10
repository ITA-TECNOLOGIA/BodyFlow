# --------------------------------------------------------------------------------
# BodyFlow
# Version: 1.0
# Copyright (c) 2023 Instituto Tecnológico de Aragón (www.itainnova.es) (Spain)
# Date: February 2023
# Authors: Ana Caren Hernández Ruiz                      ahernandez@itainnova.es
#          Ángel Gimeno Valero                              agimeno@itainnova.es
#          Carlos Marañes Nueno                            cmaranes@itainnova.es
#          Irene López Bosque                                ilopez@itainnova.es
#          María de la Vega Rodrigálvarez Chamarro   vrodrigalvarez@itainnova.es
#          Pilar Salvo Ibáñez                                psalvo@itainnova.es
#          Rafael del Hoyo Alonso                          rdelhoyo@itainnova.es
#          Rocío Aznar Gimeno                                raznar@itainnova.es
# All rights reserved 
# --------------------------------------------------------------------------------

""" 
Code for visualization of the four different 3d predictors, it does 
not work automatically. As an input it needs the four networks: mhformer, 
videopose, mixste and motionbert combined with a single 2d detector.
Usage:
    - input_video -> video path 
    - log_fnm -> a log file 
Please refer to the example bellow.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Here goes the input files
input_video = 'data/input/sample_video.mp4' 
log_fnm = 'Log_cpn_videopose_sample_video.csv'

class Visualization():
    def __init__(self, video_fnm, log_filename, alternative=True):
        """
        This function takes as an input the video and outputs a video 
        with the original video with the 2D joints and the 3D plot.  
        """
        self.video_fnm = video_fnm
        self.log_filename = log_filename
        self.pred_2d = log_filename.split('_')[1]
        self.video_name = video_fnm.split('/')[-1]
        filename = 'logs/' + log_filename
        self.df = pd.read_csv(filename)
        self.number_frames = len(self.df)
        self.alternative = alternative
        video_frames, fps = self.get_video_frames()
        self.video_frames = video_frames
        self.fps = fps
        # Get all 3d joints 
        self.df = pd.read_csv(f'logs/Log_{self.pred_2d}_videopose_{video_fnm.split("/")[-1].split(".")[0]}.csv')
        self.x1_3d = self.get3djoints()
        self.df = pd.read_csv(f'logs/Log_{self.pred_2d}_mhformer_{video_fnm.split("/")[-1].split(".")[0]}.csv')
        self.x2_3d = self.get3djoints()
        self.df = pd.read_csv(f'logs/Log_{self.pred_2d}_mixste_{video_fnm.split("/")[-1].split(".")[0]}.csv')
        self.x3_3d = self.get3djoints()
        self.df = pd.read_csv(f'logs/Log_{self.pred_2d}_motionbert_{video_fnm.split("/")[-1].split(".")[0]}.csv')
        self.x4_3d = self.get3djoints()
        self.x1_2d = self.get2djoints()
        self.plotVideo()
        

    def get_video_frames(self):
        """
        Fetchs frames of the video input with opencv and the frames per second
        """       
        import cv2
        vcap = cv2.VideoCapture(self.video_fnm)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frames = []
        ret = True
        while ret:
            ret, img = vcap.read() 
            if ret:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                frames.append(img)
        video_frames = np.stack(frames, axis=0)
        video_frames = video_frames[self.df.iloc[:,0][0]-1: self.df.iloc[:,0][len(self.df)-1]]
        print('Video frames processed')
        return video_frames, fps
    
    def get3djoints(self):
        """
        Gets the 3d keyoints froms the CSV by reading the headers. 
        """      
        joint_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                       'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
                       'right_knee', 'left_ankle', 'right_ankle', 'jaw', 'chest', 'spine', 'hips']
        axes = ['x', 'y', 'z']
        joints = np.zeros((len(self.df), len(joint_names), len(axes)))
        for i, joint in enumerate(joint_names):
            for j, ax in enumerate(axes):
                joints[:, i, j] = np.asarray(self.df.iloc[:, self.df.columns.get_loc(f'{joint}.coordinate_{ax}')])
        return joints

    def get2djoints(self):
        """
        Gets the 2d keyoints froms the CSV by reading the headers. 
        """   
        joint_names = ['hips', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 
                       'left_ankle', 'spine', 'chest', 'jaw', 'nose', 'left_shoulder', 'left_elbow', 
                       'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist']
        axes = ['x', 'y']
        joints = np.zeros((len(self.df), len(joint_names), len(axes)))
        for i, joint in enumerate(joint_names):
            for j, ax in enumerate(axes):
                joints[:, i, j] = np.asarray(self.df.iloc[:, self.df.columns.get_loc(f'bodyLandmarks2d.{joint}.coordinate_{ax}')])
        return joints   

    def update_lines(self, i):
        # Fetch 2D data of the current frame and update it
        xv1, yv1 = self.x1_2d[i][:, 0], self.x1_2d[i][:, 1]
        self.clj12d[0].set_data(xv1.take(self.l_joints2d), yv1.take(self.l_joints2d))
        self.crj12d[0].set_data(xv1.take(self.r_joints2d), yv1.take(self.r_joints2d)) 
        
        # Fetch 3D data of the current frame 
        xs1, ys1, zs1 = self.x1_3d[i][:, 0], self.x1_3d[i][:, 2], -self.x1_3d[i][:, 1]
        xs2, ys2, zs2 = self.x2_3d[i][:, 0], self.x2_3d[i][:, 2], -self.x2_3d[i][:, 1]
        xs3, ys3, zs3 = self.x3_3d[i][:, 0], self.x3_3d[i][:, 2], -self.x3_3d[i][:, 1]
        xs4, ys4, zs4 = self.x4_3d[i][:, 0], self.x4_3d[i][:, 2], -self.x4_3d[i][:, 1]
        """
        Updates the limits if alternative is True, this parameter makes your grid adapt
        to the current maximum and minimum of the joints.
        """  
        if self.alternative == True:
            new_x = np.concatenate((xs1, ys1, zs1))
            self.ax1.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_zlim3d([np.amin(new_x), np.amax(new_x)])

            new_x = np.concatenate((xs2, ys2, zs2))
            self.ax2.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_zlim3d([np.amin(new_x), np.amax(new_x)])
            
            new_x = np.concatenate((xs3, ys3, zs3))
            self.ax3.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_zlim3d([np.amin(new_x), np.amax(new_x)])

            new_x = np.concatenate((xs4, ys4, zs4))
            self.ax4.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_zlim3d([np.amin(new_x), np.amax(new_x)])
            
        # Update 3D data of the current frame
        self.clj1[0].set_data_3d(xs1.take(self.l_joints), ys1.take(self.l_joints), zs1.take(self.l_joints))
        self.crj1[0].set_data_3d(xs1.take(self.r_joints), ys1.take(self.r_joints), zs1.take(self.r_joints))
        
        self.clj2[0].set_data_3d(xs2.take(self.l_joints), ys2.take(self.l_joints), zs2.take(self.l_joints))
        self.crj2[0].set_data_3d(xs2.take(self.r_joints), ys2.take(self.r_joints), zs2.take(self.r_joints))
        
        self.clj3[0].set_data_3d(xs3.take(self.l_joints), ys3.take(self.l_joints), zs3.take(self.l_joints))
        self.crj3[0].set_data_3d(xs3.take(self.r_joints), ys3.take(self.r_joints), zs3.take(self.r_joints))
        
        self.clj4[0].set_data_3d(xs4.take(self.l_joints), ys4.take(self.l_joints), zs4.take(self.l_joints))
        self.crj4[0].set_data_3d(xs4.take(self.r_joints), ys4.take(self.r_joints), zs4.take(self.r_joints))
        
        # Update the video frame
        self.vplot.set_data(self.video_frames[i])

        return 


    def initVideo(self, log_filename):    
        i = 0
        fig = plt.figure(figsize=(11,6))
        fig.suptitle('3D Human Pose Estimation', fontsize = 18, fontweight = 'bold', y = 0.88 )
        spec = gridspec.GridSpec(ncols = 3, nrows = 3, width_ratios = [2,1, 1], 
                                 height_ratios=[.3, 1, 1], wspace = 0.2, hspace = 0.2 )
        
        
        # Logos, please keep them so more people can use our code
        logoI = plt.imread('figures/itainnova_logo.png')
        logoaxI = fig.add_axes([0.12, 0.8, 0.2, 0.2], anchor = 'NW', zorder=1 )
        logoaxI.imshow(logoI)
        logoaxI.axis('off')

        logo = plt.imread('figures/AI4HealthyAging_logo.png')
        logoax = fig.add_axes([0.7, 0.8, 0.2, 0.2], anchor = 'NE', zorder = 1 )
        logoax.imshow(logo)
        logoax.axis('off')

        logoB = plt.imread('src/pose-estimation/videos/bodyflow.png')
        logoaxB = fig.add_axes([0.5, 0.9, 0.12, 0.12], anchor = 'N', zorder = 1 )
        logoaxB.imshow(logoB)
        logoaxB.axis('off')
        
        # Video frames plot
        axv = fig.add_subplot(spec[0:,0])
        self.vplot = axv.imshow(self.video_frames[i])
        axv.set_title(f'Original Video + 2D ({log_filename.split("_")[1]})', fontweight = 'bold' )   
        axv.axis('off')
        
        # Linewidth for joints
        lw=2
        
        # Fetch and plot 2D data
        xv1, yv1 = self.x1_2d[i][:, 0], self.x1_2d[i][:, 1]
        self.clj12d = axv.plot(xv1.take(self.l_joints2d), yv1.take(self.l_joints2d), c = (0,0,1), linewidth = lw)
        self.crj12d = axv.plot(xv1.take(self.r_joints2d), yv1.take(self.r_joints2d), c = (1,0,0), linewidth = lw)   
        
        
        # Fetch 3D data (Axes are different for plotting purposes)
        xs1, ys1, zs1 = self.x1_3d[i][:, 0], self.x1_3d[i][:, 2], -self.x1_3d[i][:, 1]
        xs2, ys2, zs2 = self.x2_3d[i][:, 0], self.x2_3d[i][:, 2], -self.x2_3d[i][:, 1]
        xs3, ys3, zs3 = self.x3_3d[i][:, 0], self.x3_3d[i][:, 2], -self.x3_3d[i][:, 1]
        xs4, ys4, zs4 = self.x4_3d[i][:, 0], self.x4_3d[i][:, 2], -self.x4_3d[i][:, 1]
        
        # 3D plots 
        self.ax1 = fig.add_subplot(spec[4], projection ='3d')
        self.ax1.set_title(f'3D (Videopose3D)', fontweight='bold')
        self.ax1.set_box_aspect([1,1,1])
        self.ax1.view_init(elev = 12, azim = -60) # For better visualization
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_zticklabels([])
        
        self.ax2 = fig.add_subplot(spec[5], projection ='3d')
        self.ax2.set_title(f'3D (MHFormer)', fontweight='bold')
        self.ax2.set_box_aspect([1,1,1])
        self.ax2.view_init(elev = 12, azim = -60) # For better visualization
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_zticklabels([])

        self.ax3 = fig.add_subplot(spec[7], projection ='3d')
        self.ax3.set_title(f'3D (MixSTE)', fontweight='bold')
        self.ax3.set_box_aspect([1,1,1])
        self.ax3.view_init(elev = 12, azim = -60) # For better visualization
        self.ax3.set_xticklabels([])
        self.ax3.set_yticklabels([])
        self.ax3.set_zticklabels([])

        self.ax4 = fig.add_subplot(spec[8], projection ='3d')
        self.ax4.set_title(f'3D (MotionBert)', fontweight='bold')
        self.ax4.set_box_aspect([1,1,1])
        self.ax4.view_init(elev = 12, azim = -60) # For better visualization
        self.ax4.set_xticklabels([])
        self.ax4.set_yticklabels([])
        self.ax4.set_zticklabels([])
        
        """
        Define the limits for the grid, this parameter makes your grid adapt
        to the current maximum and minimum of the joints.
        """
        if self.alternative == True: 
            # The current min/max of the skeleton joints 
            new_x = np.concatenate((xs1, ys1, zs1)) 
            self.ax1.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_zlim3d([np.amin(new_x), np.amax(new_x)])

            new_x = np.concatenate((xs2, ys2, zs2))
            self.ax2.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_zlim3d([np.amin(new_x), np.amax(new_x)])
            
            new_x = np.concatenate((xs3, ys3, zs3))
            self.ax3.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_zlim3d([np.amin(new_x), np.amax(new_x)])

            new_x = np.concatenate((xs4, ys4, zs4))
            self.ax4.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_zlim3d([np.amin(new_x), np.amax(new_x)])
            
        else:
            # The global min/max of the whole sequence
            new_x = np.concatenate((self.x1_3d[:][0], self.x1_3d[:][2], -self.x1_3d[:][1])) 
            self.ax1.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_zlim3d([np.amin(new_x), np.amax(new_x)])   

            new_x = np.concatenate((self.x2_3d[:][0], self.x2_3d[:][2], -self.x2_3d[:][1])) 
            self.ax2.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax2.set_zlim3d([np.amin(new_x), np.amax(new_x)])
            
            new_x = np.concatenate((self.x3_3d[:][0], self.x3_3d[:][2], -self.x3_3d[:][1])) 
            self.ax3.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax3.set_zlim3d([np.amin(new_x), np.amax(new_x)])

            new_x = np.concatenate((self.x4_3d[:][0], self.x4_3d[:][2], -self.x4_3d[:][1])) 
            self.ax4.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax4.set_zlim3d([np.amin(new_x), np.amax(new_x)])   
        
        # Plot 3D joints
        self.clj1 = self.ax1.plot(xs1.take(self.l_joints), ys1.take(self.l_joints), zs1.take(self.l_joints),c = (0,0,1), linewidth = lw, zdir = 'z')
        self.crj1 = self.ax1.plot(xs1.take(self.r_joints), ys1.take(self.r_joints), zs1.take(self.r_joints), c = (1,0,0), linewidth = lw, zdir = 'z')
       
        self.clj2 = self.ax2.plot(xs2.take(self.l_joints), ys2.take(self.l_joints), zs2.take(self.l_joints),c = (0,0,1), linewidth = lw, zdir = 'z')
        self.crj2 = self.ax2.plot(xs2.take(self.r_joints), ys2.take(self.r_joints), zs2.take(self.r_joints), c = (1,0,0), linewidth = lw, zdir = 'z')

        self.clj3 = self.ax3.plot(xs3.take(self.l_joints), ys3.take(self.l_joints), zs3.take(self.l_joints),c = (0,0,1), linewidth = lw, zdir = 'z')
        self.crj3 = self.ax3.plot(xs3.take(self.r_joints), ys3.take(self.r_joints), zs3.take(self.r_joints), c = (1,0,0), linewidth = lw, zdir = 'z')

        self.clj4 = self.ax4.plot(xs4.take(self.l_joints), ys4.take(self.l_joints), zs4.take(self.l_joints),c = (0,0,1), linewidth = lw, zdir = 'z')
        self.crj4 = self.ax4.plot(xs4.take(self.r_joints), ys4.take(self.r_joints), zs4.take(self.r_joints), c = (1,0,0), linewidth = lw, zdir = 'z')
        
        print("Plotting...")
        return fig 


    def plotVideo(self):
        # Joints order for plotting for 2D and 3D sequences
        self.l_joints2d = [13, 12, 11, 8, 7, 0, 4, 5, 6]
        self.r_joints2d = [10, 9, 8, 14, 15, 16, 15, 14, 8, 7, 0, 1, 2, 3]
                
        self.l_joints = [5, 3, 1, 14, 15, 16, 7, 9, 11]
        self.r_joints = [0, 13, 14, 2, 4, 6, 4, 2, 14, 15, 16, 8, 10, 12]

        # Animation
        fig = self.initVideo(self.log_filename)
        anim = animation.FuncAnimation(fig, self.update_lines,
                                        frames = self.number_frames, interval = 1, 
                                        blit = False, repeat = False, cache_frame_data = False)
        writervideo = animation.FFMpegWriter(fps = self.fps)
        os.makedirs(os.path.join("data", "output"), exist_ok=True)
        anim.save(f'data/output/viz_4_{self.log_filename.split(".")[0][4:]}.mp4', writer = writervideo)
        print('Video saved!')
        

vis = Visualization(input_video, log_fnm)