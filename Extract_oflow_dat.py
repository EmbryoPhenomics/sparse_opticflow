# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 12:00:56 2023

@author: jmccoy
"""

def Sum_Oflow_Data(path, savepath, savename):
    
    '''
    Function to extract optic flow data and summarise into dataframes
    
    Where:
    path     = Path containing raw optic flow .csv outputs (from sparse_oflow_loop.py)
    savepath = directory to save summarised data 
    savename = name to save .csv file as
    '''

    import numpy as np
    import glob
    import os
    import pandas as pd
    
    output = pd.DataFrame(columns = ['Deployment', 'Hour', 'Datetime', 'Embryo',
                                     'cumulative_movement', 'cumulative_pos_rot', 'cumulative_neg_rot']) # create blank dataframe
    
    for filename in os.listdir(path):
        
        if filename.endswith(".csv"): 
            
            df = pd.read_csv(path + filename)  # Get data

            def split_string(input_string):
                parts = input_string.split('_')
                return parts
    
            test = split_string(filename)   # split filename into constituent parts
            
            cumulative_movement     = sum(df['center_of_mass'])             # sum each measure accross whole timepoint
            cumulative_pos_rot_d    = sum(df['positive_rotation_distance'])
            cumulative_neg_rot_d    = sum(df['negative_rotation_distance'])
    
            new_row = {'Deployment':test[0],
                       'Hour': test[1],
                       'Datetime':test[2],
                       'Embryo':test[3],
                       'cumulative_movement':cumulative_movement, 
                       'cumulative_pos_rot':cumulative_pos_rot_d, 
                       'cumulative_neg_rot':cumulative_neg_rot_d}
            new_row['Embryo'] = new_row['Embryo'].replace('.csv', '') # create new dataframe row 
    
            output.loc[len(output)] = new_row  # append row to dataframe
            
    output.to_csv(savepath + savename)
            
#%%

# Example usage

Sum_Oflow_Data('path/to/file/', 
               'path/to/save/',
               'name.csv')
