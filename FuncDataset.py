# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:57:13 2024

@author: danie
"""


import numpy as np 
from nilearn import image, masking, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import os
import CreateFuncFeatsDataset

root = 'D:\\fmri_preproc_fmap' 
atlas_dict_path = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'
N_net = 7
N_parc = 100
preproc = "preproc_bold"

functional_dataset = CreateFuncFeatsDataset.generate_Features(root, atlas_dict_path, N_net, N_parc, preproc)