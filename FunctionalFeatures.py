# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:51:01 2024

@author: danie
"""

import numpy 
import scipy 
from nilearn import image, masking, input_data
from nilearn.connectome import Con
import pandas as pd
import maptolotlib.pyplot as plt

def get_fc(ts, labels):
    fc = pd
    return fc

def get_fc_seg(ts, labels):
    
    return fc_seg

def get_fc_integ():
    
    return fc_integ:
        
        
def get_fc_stream ():
    
    return fc_stream

def get_fcs_var ():
    
    return fc_var

def get_fcd():

    return fcd

def get_falff():
    
    return falff

    
        

def getFunctionlFeatures(image, atlas) :
    
  """
        A function that extracts functional features based on fmri data and atlas parcelations:
            image: 4D fmri Data
            atlas: 3D parcelations 
            
  """
  masker = input_data.NiftiLabelsMasker(atlas, standardize=True, detrend=True, t_r=tr)
  time_series = masker.fit_transform(fmri)
  unique_labels = np.unique(atlas.get_fdata())
  
  columns = ["fc", "fc_seg", "fc_integ", "fc_stream","fcs_var" ,"fcd", "falff"]
  func_feats = pd.DataFrame(columns = columns)
  
  return func_feats
    
    

     
    
    