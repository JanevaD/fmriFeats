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
import getFF



def getFunctionalFeatures(fmri, atlas, atlas_dict):
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
    print(time_series)
    
    for i in range (time_series.shape[1]):
        variance = np.var(time_series[:,i])
        if variance < threshlold_variance:
            print("Error! Low Variance")
            
    unique_labels = np.unique(atlas.get_fdata())
    print(unique_labels)
    label_names = []
    label_names = []
    for label_num in unique_labels: 
        if label_num in atlas_dict: 
            label_names.append(atlas_dict[label_num])
        else: 
            label_names.append("Nan")
    
    time_series = pd.DataFrame(time_series, columns=label_names[1:])
    time_series = time_series.filter(like = "Networks", axis = 1)
    fc = getFF.get_fc(time_series)
    seg, integ = getFF.get_fc_seg_integ(time_series)
    fc_stream, fcs_var, fcd = getFF.get_dfc_feats(time_series)
    alff, falff = getFF.get_falff(time_series, tr = tr)

    func_feats = {
        
          'fc': fc,
          'seg': seg,
          'integ': integ,
          'fc_stream': fc_stream,
          'fcs_var': fcs_var,
          'fcd': fcd,
          'alff': alff,
          'falff': falff
          
          }
    
    return func_feats
    



                

     
    
    