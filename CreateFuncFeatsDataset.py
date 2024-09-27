# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:57:18 2024

@author: danie
"""
import os 
from nilearn import image, input_data
import pandas as pd
import plickle
import FunctionalFeatures


def generate_Features(root, N_net, N_parc, preproc, out):
    """
    
    :param root: path to dataset directory 
    :type root: str
    :param N_net: Number of Networks 
    :type N_net: int
    :param N_parc: Number of Parcellations
    :type N_parc: int
    :param preproc: fmriprep output 
    :type preproc: str
    :param out: output directory
    :type out: str


    """
     
    missing_files = []

    for sub in os.listdir(root):
    
        if sub.startswith ("sub") and len(sub) == 7:
            try:           
      
                fmri_path = os.path.join(root, sub, "ses-V0", 'func', f'{sub}_ses-V0_task-rest_run-01_space-T1w_desc-{preproc}.nii.gz')
                atlas_path = os.path.join(root, sub, "masks", f'Schaefer2018_{N_parc}Parcels_{N_net}Networks_regrid.nii.gz')
                atlas_dict_path = os.path.join('\LUTDICT',f'Schaefer_LUTS_P{N_parc}_N_{N_net}')
            
                fmri = image.load_img(fmri_path)
                atlas = image.load_img(atlas_path)
                atlas_dict = plickle.load(open('atlas_dict_path','rb'))
                
                func_feats = FunctionalFeatures.getFunctionalFeatures(fmri, atlas, atlas_dict)
                
                sub_number = int(sub.split('-')[-1]) 
                idx.append(sub_number)
                
                functional_dataset = pd.Dataframe(index = idx)
                functional_dataset.to_csv(os.path.join(out))
          
            except FileNotFoundError:
                print(f"File not found for subject {sub}, skipping...")
                missing_files.append(sub)  # Add the subject to the missing list
            except Exception as e:
                print(f"Error processing subject {sub}: {e}, skipping...")
                missing_files.append(sub)  # Add to missing list for other errors
        if missing_files:
            print(f"The following subjects had missing files: {missing_files}")
        else:
            print("All files were processed successfully.")