# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:51:01 2024

@author: danie
"""

import numpy as np
import scipy as sp
from nilearn import image, masking, input_data
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import maptolotlib.pyplot as plt
import getFF


        

def getFunctionlFeatures(fmri, atlas, atlas_dict, select):
  """
        A function that extracts functional features based on fmri data and atlas parcelations:
            image: 4D fmri Data
            atlas: 3D parcelations     
            atlas_dict: labels for different parcellations 
  """
  tr = 3.6
  threshlold_variance = 0.01
  
  masker = input_data.NiftiLabelsMasker(atlas, standardize=True, detrend=True, t_r=tr)
  time_series = masker.fit_transform(fmri)
  unique_labels = np.unique(atlas.get_fdata())
  
  label_names = []
  for label_num in unique_labels: 
      if label_num in atlas_dict: 
          label_names.append(atlas_dict[label_num])
      else: 
          label_names.append("Nan")
  
  networks = set([name.split('_')[1] for name in label_names])     
            
  time_series = pd.DataFrame(time_series, columns=label_names[1:])
  time_series = time_series.filter(like = "Networks", axis = 1)
  
  for i in range (time_series.shape[1]):
      variance = np.var(time_series[:,i])
      if variance < threshlold_variance:
          print("Error! Low Variance")
  

  
  columns = ["fc", "fc_seg", "fc_integ", "fc_stream","fcs_var" ,"fcd", "falff"]
  func_feats = pd.DataFrame(columns = columns)
    
  return func_feats
    
    

     
    
    