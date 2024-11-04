# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:57:13 2024

@author: danie
"""

import pickle
import os
import CreateFuncFeatsDatasetMNI as CreateFuncFeatsDataset


def loop_fmriprep_output(root, preproc, selected_confounds):
    

    functional_dataset = {
        "data": CreateFuncFeatsDataset.generate_Features(root,  preproc, selected_confounds)[0],
        "missing_files":  CreateFuncFeatsDataset.generate_Features(root,  preproc, selected_confounds)[1]
                    }            
   

    return functional_dataset    