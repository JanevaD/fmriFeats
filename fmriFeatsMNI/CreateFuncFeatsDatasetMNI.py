# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:57:18 2024

@author: danie
"""

class NoFeatures(Exception):
    pass

import os 
from nilearn import image
import FunctionalFeaturesMNI as FunctionalFeatures
import pandas as pd

def generate_Features(root, preproc, selected_confounds):
    """
    :param root: path to dataset directory 
    :type root: str
    :param atlas_dict_path: path to atlas dictionary
    :type atlas_dict_path: string
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
    functional_data= {}
    
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        
        if os.path.isdir(sub_path) and sub.startswith("sub"):
        
            try:           
                print(f'Processing Subject: {sub}')
                print("-"*65)
                fmri_path = os.path.join(root, sub, "ses-V0", 'func','MNI', f'{sub}_ses-V0_task-rest_run-01_space-MNI152NLin2009cAsym_desc-{preproc}.nii.gz')
                confounds_path = os.path.join(root, sub, "ses-V0", 'func', f'{sub}_ses-V0_task-rest_run-01_desc-confounds_timeseries.tsv')
                            
                fmri = image.load_img(fmri_path)
                fmri = fmri.slicer[:,:,:,1:]
                confounds_timeseries =  pd.read_csv(confounds_path, sep='\t')
                confounds = confounds_timeseries[selected_confounds]
                confounds = confounds.loc[1:]
                confounds = confounds.values
                      
                func_feats = FunctionalFeatures.getFunctionalFeatures(fmri,  confounds, sub)
                nident = int(sub.split('-')[-1]) 
                                
                functional_data[nident] = {
                    
                    'fc': func_feats['fc'],
                    'seg': func_feats['seg'],
                    'integ': func_feats['integ'],           
                    'fcs_var': func_feats['fcs_var'], 
                    'dfcs_mean_segs': func_feats['dfcs_mean_segs'],
                    'dfcs_mean_integs': func_feats['dfcs_mean_integs'],
                    'fcd_var': func_feats['fcd_var'],
                    'fcd_mean': func_feats['fcd_mean'],
                    'fcd_vars': func_feats['fcd_vars'], 
                    'fcd_means': func_feats['fcd_means'],
                    'alff': func_feats['alff'],
                    'falff': func_feats['falff'],
                    }
   
            except FunctionalFeatures.LowVarianceError as e:
                raise NoFeatures(f"No features: {e}")
                
            except FileNotFoundError:
                print(f"File not found for subject {sub}, skipping...")
                missing_files.append(sub)  
                continue
            except Exception as e:
                import traceback
                print(f"Error processing subject {sub}: {e}, skipping...")
                traceback.print_exc()
                missing_files.append(sub) 
                continue
        if missing_files:
            print(f"The following subjects had missing files: {missing_files}")
        else:
            print("All files were processed successfully.")
            
    return functional_data, missing_files