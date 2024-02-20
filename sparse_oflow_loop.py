# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:29:20 2023

@author: jmccoy
"""

def sparseoflow_loop(filepath, savepath_vid, savepath_dat):
    
    '''
    Function to apply optical flow to multiple videos in filepath
    
    Where:
    file          = filepath and file containing avi files
    savepath_vid  = directory to save optic flow video outputs 
    filename_vid  = directory to save optic flow data outputs 
    '''
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import re
    import pandas as pd
    import vuba
    import heartcv as hcv
    import sparse_oflow
    import util
    import os
    
    for filename in os.listdir(filepath):
        
        if filename.endswith(".avi"):
            
            print(filename)

            fullpath = (filepath + filename)                          # Assign filepath and filename
            replicatepath = os.path.basename(fullpath)                # for use later in writing to csv
            vid_name = os.path.splitext(replicatepath)[0] + '.avi'
            dat_name = os.path.splitext(replicatepath)[0] + '.csv'
            
            small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
            
            # Parameters ------------------------------------
            # params for Harris corner detection
            feature_params = dict(
                maxCorners=500,       # maximum number of points for motion detection
                qualityLevel=0.001,   
                minDistance=0,        # minimum distance to track between frames
                blockSize=7,          # size of neighbourhood used for corner detection
                useHarrisDetector=0,
                k=0.7,                # trade off precision and recall. Bigger k = less false corners, but miss more real corners (high precision)
                                      # smaller k means a lot more corners, so will miss less true corners, but get a lot of false ones. 
            )
            
            # Parameters for lucas kanade optical flow
            lk_params = dict(
                winSize=(120, 120), # size of search window at each pyramid level
                maxLevel=2,         # 0-based maximal pyramid level number. if set to 0, pyramids are not used, if set to 1, two levels are used etc 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            )
            
            file = filepath + filename
            split_by = 290 # acquisition length
            
            # Create the mask
            video = vuba.Video(file)
            frames = video.read(0, len(video), low_memory=False, grayscale=True)
            frames = [vuba.shrink(frame, by=0) for frame in frames]
            
            # Run analysis -----------------------------------
            writer = vuba.Writer((savepath_vid + vid_name), video, resolution=(video.width*3, video.height), codec='MJPG')
            pm, nm, ce, cm, pa, px, py, na, nx, ny = sparse_oflow.motion_tracks(frames, None, feature_params, lk_params, writer, display=False)
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
            
            ax1.plot(cm, label='Center of mass (distances)')
            ax1.legend(loc='lower right')
            
            ax2.plot(pm, label='Positive rotation (distances)')
            ax2.plot(nm, label='Negative rotation (distances)')
            ax2.legend(loc='lower right')
            
            ax3.plot(pa, label='Positive rotation (angles)')
            ax3.plot(na, label='Negative rotation (angles)')
            ax3.legend(loc='lower right')
            
            ax4.plot(px, label='Positive rotation (x)')
            ax4.plot(nx, label='Negative rotation (x)')
            ax4.plot(py, label='Positive rotation (y)')
            ax4.plot(ny, label='Negative rotation (y)')
            ax4.legend(loc='lower right')
            
            # Save output to csv
            dat_summary = pd.DataFrame(
                {'center_of_mass': cm,
                 'positive_rotation_distance': pm,
                 'negative_rotation_distance': nm,
                 'positive_rotation_angles': pa,
                 'negative_rotation_angles': na,
                 'positive_rotation_x': px,
                 'negative_rotation_x': nx,
                 'positive_rotation_y': py,
                 'negative_rotation_y': ny
                 })
            
            dat_summary.to_csv(savepath_dat + dat_name)
            
            plt.show()
            
            video.close()
            
#%%

# Example usage

sparseoflow_loop('path/to/videos/', 
                 'path/to/save/data/', 
                 'path/to/save/video/outputs/')
