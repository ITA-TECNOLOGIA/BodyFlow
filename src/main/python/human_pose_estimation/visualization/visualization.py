# --------------------------------------------------------------------------------
# BodyFlow
# Version: 2.0
# Copyright (c) 2024 Instituto Tecnologico de Aragon (www.ita.es) (Spain)
# Date: March 2024
# Authors: Ana Caren Hernandez Ruiz                      ahernandez@ita.es
#          Angel Gimeno Valero                              agimeno@ita.es
#          Carlos Maranes Nueno                            cmaranes@ita.es
#          Irene Lopez Bosque                                ilopez@ita.es
#          Jose Ignacio Calvo Callejo                       jicalvo@ita.es
#          Maria de la Vega Rodrigalvarez Chamarro   vrodrigalvarez@ita.es
#          Pilar Salvo Ibanez                                psalvo@ita.es
#          Rafael del Hoyo Alonso                          rdelhoyo@ita.es
#          Rocio Aznar Gimeno                                raznar@ita.es
#          Pablo Perez Lazaro                               plazaro@ita.es
#          Marcos Marina Castello                           mmarina@ita.es
# All rights reserved 
# --------------------------------------------------------------------------------

#  Code for visualization of a single output
import cv2
import logging
import numpy as np
import os
import pandas as pd
import pandas.io.common
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
from matplotlib import gridspec
from human_pose_estimation.visualization.visualization_utils import get_frames#from human_pose_estimation.visualization.visualization_utils import get_frames  from visualization_utils import get_frames

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Visualization():
    def __init__(self, path, input_type, log_filename, person_id, alternative=False,output_path='.',logo1=None,logo2=None,logo3=None, timestamp = '', devs=False):
        """
        Initializes the object with the given parameters.
        
        Args:
            path (str): The path to the input video or pictures.
            input_type (str): The type of input, either 'video' or 'pictures'.
            log_filename (str): The filename of the log file.
            person_id (int): The ID of the person to process.
            alternative (bool, optional): Whether to use alternative mode. Defaults to True.
            output_path (str, optional): The path to save the output files. Defaults to '.'.
            logo1 (None, optional): The first logo image. Defaults to None.
            logo2 (None, optional): The second logo image. Defaults to None.
            logo3 (None, optional): The third logo image. Defaults to None.
            timestamp (str, optional): The timestamp. Defaults to an empty string.
            devs (bool, optional): Whether to include developer mode. Defaults to False.
        
        Raises:
            AssertionError: If the input_type is not 'video' or 'pictures'.
        
        Returns:
            None
        """
        assert input_type in ['video', 'pictures'], f"Visualization not available for input type {input_type}"

        self.path = path
        self.log_filename = log_filename
        self.person_id = person_id
        self.logo1 = logo1
        self.logo2 = logo2
        self.logo3 = logo3
        self.timestamp = timestamp
        self.devs = devs
        self.output_path = output_path

        self.left_color = (1,0,0)
        self.right_color = (0,0,1)
        self.center_color = 'darkorange'
        self.vis_aux = os.path.join("data", "vis_aux_" + self.timestamp)
        os.makedirs(self.vis_aux, exist_ok=True)
        
        try:
            self.df = pd.read_csv(self.log_filename)
            self.df = self.df[self.df['id'] == self.person_id]
            self.number_frames = len(self.df)
            self.alternative = alternative
            self.images_path, self.fps = get_frames(path, input_type, self.vis_aux)
            self.x1_3d, _ = self.get_joints(dims='3d')
            self.x1_2d, self.valid_keypoints2d = self.get_joints(dims='2d')
            if 'mediapipe3d' in self.log_filename:
                self.x2_3d = self.getExtra3djoints()
                self.x2_2d = self.getExtra2djoints()
            self.plotVideo()
        except pandas.errors.EmptyDataError:
            logging.info("CSV empty: The video was processed, but no person was recognized in the video.")
        
    
    def get_joints(self, dims = '2d'):
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        :param dims: A string indicating the dimensionality of the joints. Default is '2d'.
        :type dims: str

        :return: A tuple containing the joints and the valid keypoints.
            - joints: A numpy array representing the joint positions.
            - valid_keypoints: A list of valid keypoints.
        :rtype: tuple
        """
        joint_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle', 'jaw', 'chest', 'spine', 'hips',
                    'center_of_mass']
        if dims == '3d':
            axes = ['x', 'y', 'z']
            names = ''
            valid_keypoints = None
        elif dims =='2d':
            axes = ['x', 'y']
            names = 'bodyLandmarks2d.'
            valid_keypoints = self.df.iloc[:, self.df.columns.get_loc(f'bodyLandmarks2d.repeated')].tolist()
        joints = np.zeros((len(self.df), len(joint_names), len(axes)))

        for i, joint in enumerate(joint_names):
            for j, ax in enumerate(axes):
                column_name = f'{names}{joint}.coordinate_{ax}'
                joints[:, i, j] = self.df[column_name].to_numpy()

        return joints, valid_keypoints          
    
    
    def getExtra3djoints(self):   
        """
        Retrieves the extra 3D joints from the dataset.

        Returns:
            numpy.ndarray: A 3D array containing the coordinates of the extra joints for each frame.
                           The shape of the array is (n_frames, n_joints, n_axes), where:
                           - n_frames: The number of frames in the dataset.
                           - n_joints: The number of extra joints.
                           - n_axes: The number of axes (x, y, z) for each joint.
        """
        joint_names = ['left_ankle', 'left_heel', 'left_foot_index',
                       'right_ankle', 'right_heel', 'right_foot_index',
                       'left_ear', 'left_eye_outer', 'left_eye', 'left_eye_inner', 'nose',
                       'right_eye_inner', 'right_eye', 'right_eye_outer',  'right_ear',
                       'mouth_left', 'mouth_right'
                       ]
        axes = ['x', 'y', 'z']
        joints = np.zeros((len(self.df), len(joint_names), len(axes)))
        for i, joint in enumerate(joint_names):
            for j, ax in enumerate(axes):
                joints[:, i, j] = np.asarray(self.df.iloc[:, self.df.columns.get_loc(f'{joint}.coordinate_{ax}')])
        return joints


    def getExtra2djoints(self):
        """
        Retrieves the 2D coordinates of additional joints from the data frame.

        Returns:
            numpy.ndarray: A 3D array containing the 2D coordinates of the additional joints. 
                The shape of the array is (N, M, 2), where N is the number of rows in the data frame, 
                M is the number of additional joints, and 2 represents the x and y coordinates.
        """
        joint_names = ['left_ankle', 'left_heel', 'left_foot_index',
                       'right_ankle', 'right_heel', 'right_foot_index'
                       ]
        axes = ['x', 'y']
        joints = np.zeros((len(self.df), len(joint_names), len(axes)))
        for i, joint in enumerate(joint_names):
            for j, ax in enumerate(axes):
                joints[:, i, j] = np.asarray(self.df.iloc[:, self.df.columns.get_loc(f'bodyLandmarks2d.{joint}.coordinate_{ax}')])
        return joints   



    def plot_frames(self, i):    
        """
        Generates a plot of the frames for a given index.

        Parameters:
        - i (int): The index of the frame to plot.

        Returns:
        - None
        """
        fig = plt.figure(figsize=(10,6))
        margins = {  #     vvv margin in inches
            "left"   :     .05,
            "bottom" :    .05,
            "right"  : 1-.05,
            "top"    :1-.05
        }
        fig.subplots_adjust(**margins)
        if self.devs == True:
            fig.suptitle('BodyFlow. ID: %d' %self.person_id, fontsize = 18, fontweight = 'bold', y = 0.95 )
        spec = gridspec.GridSpec(ncols = 2, nrows = 2, width_ratios = [1.8, 1], height_ratios = [2, 0.5],
                                wspace = 0.1, hspace = 0.1 )
        
        # Logos, please keep them so more people can use our code
        if self.logo1 is not None:
            try:                
                logo1 = plt.imread(self.logo1)
                logoax1 = fig.add_axes([0.1, 0.1, 0.18, 0.18],  zorder = 10 )
                logoax1.imshow(logo1)
                logoax1.axis('off')
            except :
                logging.info('logo1 not found')
                pass

        if self.logo2 is not None:
            try:                
                logo2 = plt.imread(self.logo2)
                logoax2 = fig.add_axes([0.4, 0.1, 0.18, 0.18], zorder = 10 )
                logoax2.imshow(logo2)
                logoax2.axis('off')
            except :
                logging.info('logo2 not found')
                pass
        if self.logo3 is not None:
            try:                
                logo3 = plt.imread(self.logo3)#ita
                logoax3 = fig.add_axes([0.7, 0.1, 0.18, 0.18], zorder = 10 )
                logoax3.imshow(logo3)
                logoax3.axis('off')
            except :
                logging.info('logo3 not found')
                pass

        # Video frames plot
        axv = fig.add_subplot(spec[0])
        
        img = cv2.imread(self.images_path[self.df.iloc[i]['timestamp']])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.vplot = axv.imshow(img)

        axv.axis('off')
        
        # Linewidth for joints
        lw=2
        
        # Fetch and plot 2D data
        xv1, yv1 = self.x1_2d[i][:, 0], self.x1_2d[i][:, 1]
        self.clj12d = axv.plot(xv1.take(self.l_joints), yv1.take(self.l_joints), c = self.left_color, linewidth = lw)
        self.crj12d = axv.plot(xv1.take(self.r_joints), yv1.take(self.r_joints), c = self.right_color, linewidth = lw)   
        self.mass2d = axv.scatter(*[self.x1_2d[i][17, 0], self.x1_2d[i][17, 1]], c='cyan', marker='o', s = 50)
        
        # Fetch 3D data (Axes are different for plotting purposes)
        xs1, ys1, zs1= self.x1_3d[i][:, 0], self.x1_3d[i][:, 2], -self.x1_3d[i][:, 1]
        
        # 3D plots 
        self.ax1=fig.add_subplot(spec[1], projection='3d')
        if self.devs == True:
            axv.set_title(f'Original Video + 2D ({self.log_filename.split("_")[1]})', fontweight = 'bold' )
            self.ax1.set_title(f'3D ({self.log_filename.split("_")[2]})', fontweight = 'bold')
        
        self.ax1.set_box_aspect([1,1,1])
        self.ax1.view_init(elev = 12, azim = -60) # For better visualization
        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax1.set_zticklabels([])

        """
        Define the limits for the grid, this parameter makes your grid adapt
        to the current maximum and minimum of the joints.
        """
        if self.alternative == True and 'mediapipe3d' not in self.log_filename: 
            # The current min/max of the skeleton joints 
            new_x = np.concatenate((xs1, ys1, zs1)) 
            self.ax1.set_xlim3d([np.min(xs1), np.max(xs1)])
            self.ax1.set_ylim3d([np.min(ys1), np.max(ys1)])
            self.ax1.set_zlim3d([np.min(zs1), np.max(zs1)])
        elif self.alternative == True and 'mediapipe3d' in self.log_filename:
            self.ax1.set_xlim3d([np.nanmin(np.concatenate((self.x1_3d[i][:, 0], self.x2_3d[i][:, 0]))), 
                                 np.nanmax(np.concatenate((self.x1_3d[i][:, 0], self.x2_3d[i][:, 0])))])
            self.ax1.set_ylim3d([np.nanmin(np.concatenate((self.x1_3d[i][:, 2], self.x2_3d[i][:, 2]))), 
                                 np.nanmax(np.concatenate((self.x1_3d[i][:, 2], self.x2_3d[i][:, 2])))])
            self.ax1.set_zlim3d([np.nanmin(np.concatenate((-self.x1_3d[i][:, 1], -self.x2_3d[i][:, 1]))),
                                 np.nanmax(np.concatenate((-self.x1_3d[i][:, 1], -self.x2_3d[i][:, 1])))])     
         
        elif self.alternative == False and 'mediapipe3d' in self.log_filename:
            self.ax1.set_xlim3d([np.nanmin((np.nanmin(self.x1_3d[:,:,0]), np.nanmin(self.x2_3d[:,:,0]))), 
                                 np.nanmax((np.nanmax(self.x1_3d[:,:,0]), np.nanmax(self.x2_3d[:,:,0])))])
            self.ax1.set_ylim3d([np.nanmin((np.nanmin(self.x1_3d[:,:,2]), np.nanmin(self.x2_3d[:,:,2]))), 
                                 np.nanmax((np.nanmax(self.x1_3d[:,:,2]), np.nanmax(self.x2_3d[:,:,2])))])
            self.ax1.set_zlim3d([np.nanmin((np.nanmin(-self.x1_3d[:,:,1]), np.nanmin(-self.x2_3d[:,:,1]))), 
                                 np.nanmax((np.nanmax(-self.x1_3d[:,:,1]), np.nanmax(-self.x2_3d[:,:,1])))])  
         
                    
        else:
            # The global min/max of the whole sequence
            new_x = np.concatenate((self.x1_3d[:][0], self.x1_3d[:][2], -self.x1_3d[:][1]))
            self.ax1.set_xlim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_ylim3d([np.amin(new_x), np.amax(new_x)])
            self.ax1.set_zlim3d([np.amin(new_x), np.amax(new_x)])      
        
        # Plot 3D joints
        self.clj1 = self.ax1.plot(xs1.take(self.l_joints), ys1.take(self.l_joints), zs1.take(self.l_joints),c = self.left_color, linewidth = lw, zdir = 'z')
        self.crj1 = self.ax1.plot(xs1.take(self.r_joints), ys1.take(self.r_joints), zs1.take(self.r_joints), c = self.right_color, linewidth = lw, zdir = 'z')
        self.mass3d = self.ax1.scatter(*[xs1[17], ys1[17], zs1[17]], c= self.center_color, marker='o', s=50)

        if 'mediapipe3d' in self.log_filename:
            self.left_feet = [0, 1, 2, 0]
            self.right_feet = [3, 4, 5, 3]            
            self.eyes = [6, 7, 8, 9, 10, 11, 12, 13, 14]
            self.mouth = [15, 16]
            
            # Fetch and plot 2D data
            xv12, yv12 = self.x2_2d[i][:, 0], self.x2_2d[i][:, 1]
            self.clj22d = axv.plot(xv12.take(self.left_feet), yv12.take(self.left_feet), c = self.left_color, linewidth = lw)
            self.crj22d = axv.plot(xv12.take(self.right_feet), yv12.take(self.right_feet), c = self.right_color, linewidth = lw)   
            
            # Fetch 3D data (Axes are different for plotting purposes)
            xs2, ys2, zs2= self.x2_3d[i][:, 0], self.x2_3d[i][:, 2], -self.x2_3d[i][:, 1]
            

            # Plot 3D joints
            self.clj12 = self.ax1.plot(xs2.take(self.left_feet), ys2.take(self.left_feet), zs2.take(self.left_feet),c = self.left_color, linewidth = lw, zdir = 'z')
            self.crj12 = self.ax1.plot(xs2.take(self.right_feet), ys2.take(self.right_feet), zs2.take(self.right_feet), c = self.right_color, linewidth = lw, zdir = 'z')
            self.eyes3d = self.ax1.plot(xs2.take(self.eyes), ys2.take(self.eyes), zs2.take(self.eyes), c = self.right_color, linewidth = lw, zdir = 'z')
            self.mouth3d = self.ax1.plot(xs2.take(self.mouth), ys2.take(self.mouth), zs2.take(self.mouth), c = self.right_color, linewidth = lw, zdir = 'z')
        fig.savefig(os.path.join(self.vis_out, 'viz_%06d.png' % i))
        plt.close(fig)

    def plotVideo(self):
        """
        Plot the video frames and save the result as an mp4 file.

        Parameters:
            None

        Returns:
            None
        """
        # Joints order for plotting for 2D and 3D sequences
        self.l_joints = [5, 3, 1, 14, 15, 16, 7, 9, 11]
        self.r_joints = [0, 13, 14, 2, 4, 6, 4, 2, 14, 15, 16, 8, 10, 12]

        resolutions = {
            "480p": "720x480",
            "576p": "720x576",
            "720p": "1280x720",
            "1080p": "1920x1080",
            "1440p": "2560x1440",
            "2K": "2048x1080",
            "4K": "3840x2160",
            "8K": "7680x4320"
        }
        output_resolution='720p'

        # Animation
        self.vis_out = os.path.join("data", "vis_out_" + self.timestamp)
        os.makedirs(self.vis_out, exist_ok = True)
        
        for i in range(self.number_frames):
            logging.info(f'Plotting frame {i} /{self.number_frames}')
            self.plot_frames(i)
            
        if self.devs == True:
            video_name = self.log_filename.split('/')[-1].split(".")[0][4:]
        else:
            video_name = self.path.split('/')[-1].split('.')[0]

        try:
            command = f"ffmpeg -r {self.fps} -f image2 -s " + resolutions[output_resolution] + f" -i {self.vis_out}//viz_%06d.png -vcodec libx264 -crf 25  {self.output_path}\\viz_{video_name}.mp4 -y"
            os.system(command)
            logging.info('Video saved!')
        except Exception as e:
            logging.info(f"An error occurred while saving the visualization: {e}")
        
        
        # Remove all unnecessary files
        for file in os.listdir(self.vis_aux):
            file_path = os.path.join(self.vis_aux, file)
            if os.path.isfile(file_path):
                os.remove(file_path)   
                
        for file in os.listdir(self.vis_out):
            file_path = os.path.join(self.vis_out, file)
            if os.path.isfile(file_path):
                os.remove(file_path)    
        
        os.rmdir(self.vis_aux)
        os.rmdir(self.vis_out)
        
        logging.info('Removed files')
        

if __name__ == "__main__":
    path = 'data/input/cinta_markers_007_iphone.MOV'
    input ='video'
    pose_logger = 'logs/Log_cpn_motionbert_cinta_markers_007_iphone.csv'
    output_path = ''
    logo1 = 'figures/ITA_Logo.png',
    logo2 = 'figures/AI4HealthyAging_logo.png',
    logo3 = 'figures/ITA_Logo.png',
    timestamp = ''
    Visualization(path, input, pose_logger, person_id = 1, alternative= False, output_path = output_path, logo1 = logo1, logo2 = logo2, logo3 = logo3, timestamp = timestamp, devs = True)