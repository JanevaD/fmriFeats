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

def get_fc(time_series):
    
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series.values])[0]
    
    fc = pd.DataFrame(correlation_matrix, columns= time_series.columns, index = time_series.columns)
    names = ['_'.join(name.split('_')[1:3]) for name in fc.columns]
    fc.columns = names
    fc.index = names
    
    return fc

def get_fc_seg_integ(fc, networks):
    
    seg_fcs =[]
    integ_fcs=[]
    for network in networks:        
        columns = [col for col in fc.columns if f'{network}_' in col]
        num_columns = len(columns)
            
        seg_fc = fc.loc[columns,columns].sum().sum()
        seg_fcs.append(seg_fc/num_columns)
        
        integ_fc = fc.copy()
        integ_fc.loc[columns,columns]=0
        
        integ_fc = integ_fc.loc[columns,::].sum().sum()
        integ_fcs.append(integ_fc/(integ_fc.shape[1]-num_columns))
             
    return seg_fcs, integ_fcs   

#def dim(corrs,k):
 #   "k is the number of windows"
  #  merged_matrices = []
   # for i in range(0, len(corrs),k):
    #    if i+k<=len(corrs):
     #       merged = np.stack(corrs[i:i+k],axis=-1)
      #      merged_matrices.append(merged)

    #final_array = np.stack(merged_matrices, axis=0)

    #return final_array

def get_dfc (time_series,M,L,S):
    """
        M is timeseries length 
        L is windows length 
        S is step size 
        
    """
    fc_stream =[]
    for i in range (0, M-S, S):
        correlation_measure = ConnectivityMeasure(kind='correlation')
        dfc = correlation_measure.fit_transform([time_series[i:i+L].values])[0]
        fc_stream.append(dfc)        
        
   # fc_stream = dim(fc_stream, int(np.ceil((len(time_series)-l)/s)+1 ))
    fcs_var = np.var(fc_stream)
    fcd = np.corrcoef(fcs_var)
    
    return fc_stream, fcs_var, fcd



def get_falff():
    falff = []
    return falff


    
        

def getFunctionlFeatures(fmri, atlas, atlas_dict):
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
    
    

     
    
    