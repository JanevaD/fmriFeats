# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:57:18 2024

@author: danie
"""
import os 
from nilearn import image
import FunctionalFeatures


def generate_Features(root, atlas_dict, N_net, N_parc, preproc):
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
    idx = []
    functional_data= {}
    
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        
        if os.path.isdir(sub_path) and sub.startswith("sub"):
        
            print(f"Processing folder: {sub}")
            print("-"*65)
            try:           
                fmri_path = os.path.join(root, sub, "ses-V0", 'func', f'{sub}_ses-V0_task-rest_run-01_space-T1w_desc-{preproc}.nii.gz')
                print(fmri_path)
                print("-"*65)
                atlas_path = os.path.join(root, sub, "masks", f'Schaefer2018_{N_parc}Parcels_{N_net}Networks_regrid.nii.gz')
                print(atlas_path)
                print("-"*65)
     
                            
                fmri = image.load_img(fmri_path)
                atlas = image.load_img(atlas_path)
                print(atlas.shape, fmri.shape )

                      
                func_feats = FunctionalFeatures.getFunctionalFeatures(fmri, atlas, atlas_dict)
       
                
                
                nident = int(sub.split('-')[-1]) 
                
                
                
                functional_data[nident] = {
                    'fc': func_feats['fc'],
                    'seg': func_feats['seg'],
                    'integ': func_feats['integ'],
                    'fc_stream': func_feats['fc_stream'],
                    'fcs_var': func_feats['fcs_var'],
                    'fcd': func_feats['fcd'],
                    'alff': func_feats['alff'],
                    'falff': func_feats['falff'],
                    }
                
                print(functional_data)
                print("-"*65)
                
                
            except FileNotFoundError:
                print(f"File not found for subject {sub}, skipping...")
                missing_files.append(sub)  
            except Exception as e:
                import traceback
                print(f"Error processing subject {sub}: {e}, skipping...")
                traceback.print_exc()
                missing_files.append(sub) 
        if missing_files:
            print(f"The following subjects had missing files: {missing_files}")
        else:
            print("All files were processed successfully.")
            
    return functional_data