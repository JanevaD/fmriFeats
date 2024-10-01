# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:57:13 2024

@author: danie
"""


 
import pickle
import os
import CreateFuncFeatsDataset


def loop_fmriprep_output(root,atlas_dict_path, P_N, preprocessing):
    
    functional_datasets = []
    for N_parc, N_net in P_N: 
        atlas_dict_path = os.path.join(atlas_dict_path,f'Schaefer_LUTS_P{N_parc}_N{N_net}.pkl')
    
        with open(atlas_dict_path, 'rb') as f:
            atlas_dict = pickle.load(f)
       
        for preproc in preprocessing: 
            print(f'Processing functional files P={N_parc}, N={N_net}, preprocessing={preproc}:')
            print ('='*65)
            try:
                functional_dataset = {
                    "Networks": N_net,
                    "Parcellations": N_parc,
                    "preprocessing": preproc,
                    "data": CreateFuncFeatsDataset.generate_Features(root, atlas_dict, N_net, N_parc, preproc)[0],
                    "missing_files":  CreateFuncFeatsDataset.generate_Features(root, atlas_dict, N_net, N_parc, preproc)[1]
                    }
                functional_datasets.append(functional_dataset)
                
            except CreateFuncFeatsDataset.NoFeatures as e: 
                print(f"Can't generate feaures for desc-{preproc} output: {e} ")
                continue
               
            except Exception as e:
                print(f"Error processing P={N_parc}, N={N_net}, preprocessing={preproc}: {e}")
           
            return functional_datasets    
         
    
         
    


