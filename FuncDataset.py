# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:57:13 2024

@author: danie
"""


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
from nilearn import image, masking, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import os
import CreateFuncFeatsDataset
import LoopFmriprepOutput


#%%

root = 'D:\\fmri_preproc_fmap' 
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'

preprocessing = ["AROMA_bold_gm",  "preproc_bold_masked_gm_intens", 
                 "aparcaseg_dseg",  "preproc_bold", "AROMA_bold", "brain_mask", "aseg_dseg", 
                 "AROMAnonaggr_denoised_ROI", "AROMA_bold_gm_intens", "preproc_bold_masked_gm"]

P_N = [[100, 7],[200, 7],[100, 17],[500, 17],[1000, 7],[1000, 17]]

functional_datasets = LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing)

#%%
root = 'D:\\fmri_preproc_fmap' 
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'

preprocessing = [ 'AROMA_bold_no_mask']

P_N = [[100,7],[200,17]]

functional_datasets = LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing)
         
       

#%%
root = 'D:\\fmri_preproc_fmap' 
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'

preprocessing = [ 'AROMA_bold_no_mask']

P_N = [[100, 7],[200, 7],[100, 17],[500, 17],[1000, 7],[1000, 17]]

functional_datasets = LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing)
         
       
#%%
results = pd.read_csv('C:/Users/danie/phd/pharmo-fmri/results 30/Results k=2.csv')

#%% 
for dataset in  functional_datasets:
    
    N_n = dataset["Networks"]
    N_p = dataset["Parcellations"]
    preproc = dataset["preprocessing"]
    
    print(N_n, N_p)
    print("="*65)

    nident = []
    seg = pd.DataFrame()
    integ = pd.DataFrame() 
    for i, key in enumerate(dataset["data"].keys()):
        
        nident.append(key)

        seg = pd.concat([seg,dataset['data'][key]['seg'].T],axis =1)
        integ = pd.concat([integ,dataset['data'][key]['integ']],axis = 1)
        networks = dataset['data'][key]['seg'].index
       
    seg = seg.T
    integ = integ.T    
    
    
    seg.reset_index(drop = True, inplace = True); seg.set_index(np.array(nident), inplace = True)
    integ.reset_index(drop = True, inplace = True); integ.set_index(np.array(nident), inplace = True)
    
    print (seg.head())
    print("-"*65)
    print(integ.head())
    print("-"*65)

    seg_class = seg.merge(results[['Clusters']], left_index=True, right_index=True, how = 'inner')
    integ_class = integ.merge(results[['Clusters']], left_index=True, right_index = True, how ='inner')

    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.25)
    Q3 = SegInteg['Value'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR

    filtered_data = SegInteg[(SegInteg['Value'] >= lower_bound) & (SegInteg['Value'] <= upper_bound)]

    fig, axes = plt.subplots(3,1, figsize = (16,8))

    colors = ['#FF0101','#0E0EFF']

    sns.stripplot(
        x='Feature', y='Value', hue='Clusters',
      #  kind='strip',  
        data = features_seg,
        palette=colors,
        dodge = True,
        size = 4,
        alpha = 0.6,
        ax=axes[0],
    )

    sns.boxplot(
        x='Feature', y='Value', hue='Clusters',
        data = features_seg,
        palette=colors,
     #   gap = 0.5,
        dodge=True,
        boxprops=dict(facecolor="none") , 
      #  width = 0.3,
        ax=axes[0],
    )

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]

    axes[0].set_title('Segregated FC by Cluster')

    sns.stripplot(
        x='Feature', y='Value', hue='Clusters',
        #kind='strip',  
        data=features_integ,
        palette=colors,
        ax=axes[1],
        size = 3,
        alpha = 0.6,
        dodge = True,
        #legend=False
    )

    sns.boxplot(
        x='Feature', y='Value', hue='Clusters',
        data=features_integ,
        palette=colors,
        boxprops=dict(facecolor="none")  ,
        dodge = True,
        ax = axes[1],
       # width = 0.3,
        #gap = 0.4,    
    )

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]

    axes[1].set_title('Integrated FC by Cluster')

    sns.stripplot(
        x='Feature', y='Value', hue='Clusters',
        #kind='strip',  
        data=filtered_data,
        palette=colors,
        ax=axes[2],
        size = 3,
        alpha = 0.6,
        dodge = True,
        #legend=False
    )

    sns.boxplot(
        x='Feature', y='Value', hue='Clusters',
        data=filtered_data,
        palette=colors,
        boxprops=dict(facecolor="none")  ,
        dodge = True,
        ax = axes[2],
        showfliers=False
       # width = 0.3,
        #gap = 0.4,    
    )

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
    axes[2].set_title('Ratio Seg/Integ')


    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc}")
    plt.tight_layout()  
    plt.show()

#%%
for dataset in  functional_datasets:
    
    N_n = dataset["Networks"]
    N_p = dataset["Parcellations"]
    preproc = dataset["preprocessing"]
    
    print(N_n, N_p)
    print("="*65)

    nident = []
    seg = pd.DataFrame()
    integ = pd.DataFrame() 
    for i, key in enumerate(dataset["data"].keys()):
        
        nident.append(key)

        seg = pd.concat([seg,pd.DataFrame(dataset['data'][key]['falff'])],axis =1)
        integ = pd.concat([integ,pd.DataFrame(dataset['data'][key]['alff'])],axis = 1)
        networks = dataset['data'][key]['fc'].index
       
    seg = seg.T
    integ = integ.T    
    seg.columns = networks; integ.columns = networks;
    
    seg= seg.groupby(seg.columns, axis=1).mean()
    integ= integ.groupby(integ.columns, axis=1).mean()

    
    seg.reset_index(drop = True, inplace = True); seg.set_index(np.array(nident), inplace = True)
    integ.reset_index(drop = True, inplace = True); integ.set_index(np.array(nident), inplace = True)
    
    
    print (seg.head())
    print("-"*65)
    print(integ.head())
    print("-"*65)

    seg_class = seg.merge(results[['Clusters']], left_index=True, right_index=True, how = 'inner')
    integ_class = integ.merge(results[['Clusters']], left_index=True, right_index = True, how ='inner')

    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.25)
    Q3 = SegInteg['Value'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR

    filtered_data = SegInteg[(SegInteg['Value'] >= lower_bound) & (SegInteg['Value'] <= upper_bound)]

    fig, axes = plt.subplots(2,1, figsize = (16,8))

    colors = ['#FF0101','#0E0EFF']

    sns.stripplot(
        x='Feature', y='Value', hue='Clusters',
      #  kind='strip',  
        data = features_seg,
        palette=colors,
        dodge = True,
        size = 4,
        alpha = 0.6,
        ax=axes[0],
    )

    sns.boxplot(
        x='Feature', y='Value', hue='Clusters',
        data = features_seg,
        palette=colors,
     #   gap = 0.5,
        dodge=True,
        boxprops=dict(facecolor="none") , 
      #  width = 0.3,
        ax=axes[0],
    )

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[0].get_xticklabels()]

    axes[0].set_title('Falff by Cluster')

    sns.stripplot(
        x='Feature', y='Value', hue='Clusters',
        #kind='strip',  
        data=features_integ,
        palette=colors,
        ax=axes[1],
        size = 3,
        alpha = 0.6,
        dodge = True,
        #legend=False
    )

    sns.boxplot(
        x='Feature', y='Value', hue='Clusters',
        data=features_integ,
        palette=colors,
        boxprops=dict(facecolor="none")  ,
        dodge = True,
        ax = axes[1],
       # width = 0.3,
        #gap = 0.4,    
    )

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]

    axes[1].set_title('Alff by Cluster')




    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc}")
    plt.tight_layout()  
    plt.show()
