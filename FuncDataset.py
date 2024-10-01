# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:57:13 2024

@author: danie
"""


import numpy as np 
import pickle
from nilearn import image, masking, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import os
import CreateFuncFeatsDataset

root = 'D:\\fmri_preproc_fmap' 
atlas_dict_path = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'
#%%
N_net = 7
N_parc = 100
preproc = "preproc_bold"

atlas_dict_path = os.path.join(atlas_dict_path,f'Schaefer_LUTS_P{N_parc}_N{N_net}.pkl')

with open(atlas_dict_path, 'rb') as f:
    atlas_dict = pickle.load(f)
    
functional_dataset = {
    "Networks": N_net,
    "Parcellations": N_parc,
    "preprocessing": preproc,
    "data": CreateFuncFeatsDataset.generate_Features(root, atlas_dict, N_net, N_parc, preproc)
}

#%%

preprocessing = ["AROMA_bold_gm", "MELODIC_mixing", "preproc_bold_z_scored", "preproc_bold_masked_gm_intens", 
                 "aparcaseg_dseg", "confounds_timeseries", "preproc_bold", "AROMA_bold", "brain_mask", "aseg_dseg", 
                 "AROMAnonaggr_denoised_ROI", "AROMA_bold_gm_intens", "preproc_bold_masked_gm"]

P_N = [[100, 7],[200, 7],[100, 17],[500, 17],[1000, 7],[1000, 17]]

functional_datasets = []
for N_parc, N_net in P_N: 
    atlas_dict_path = os.path.join(atlas_dict_path,f'Schaefer_LUTS_P{N_parc}_N{N_net}.pkl')
    
    try:
        with open(atlas_dict_path, 'rb') as f:
            atlas_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Atlas dictionary for P={N_parc} and N={N_net} not found, skipping...")
        continue
    
    try:     
        for preproc in preprocessing: 
            print(f'Processing functional files P={N_parc}, N={N_net}, preprocessing={preproc}:')
            print ('-'*65)
            functional_dataset = {
                "Networks": N_net,
                "Parcellations": N_parc,
                "preprocessing": preproc,
                "data": CreateFuncFeatsDataset.generate_Features(root, atlas_dict, N_net, N_parc, preproc)
                }
            functional_datasets.append(functional_dataset)
    except Exception as e:
         print(f"Error processing P={N_parc}, N={N_net}, preprocessing={preproc}: {e}")
     


