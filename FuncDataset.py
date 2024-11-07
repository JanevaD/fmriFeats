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
#import CreateFuncFeatsDataset
#import LoopFmriprepOutput
import scipy.stats as st


#%%

root = 'D:\\fmri_preproc_fmap' 
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'


preprocessing = ["AROMA_bold_gm",  "preproc_bold_masked_gm_intens", 
                 "aparcaseg_dseg",  "preproc_bold", "AROMA_bold", "brain_mask", "aseg_dseg", 
                 "AROMAnonaggr_denoised_ROI", "AROMA_bold_gm_intens", "preproc_bold_masked_gm"]

P_N = [[100, 7],[200, 7],[100, 17],[500, 17],[1000, 7],[1000, 17]]

functional_datasets = LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing)
#%%
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\fromgithub\\fmriFeats\\LUTDict'
results = pd.read_csv('C:\\Users\\danie\\phd\\pharmo-fmri\\fromgithub\\fmriFeats\\Results k=2.csv')


#%%
root = 'D:\\fmri_preproc_fmap' 

atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'
#%%

preprocessing = ['AROMA_bold_no_mask']

confound_vars = ['global_signal', 'csf', 'white_matter']

derivative_columns = ['{}_derivative1'.format(c) for c in confound_vars]

confound_vars_power2 = ['{}_power2'.format(c) for c in confound_vars]

derivative_power2 =  ['{}_power2'.format(c) for c in derivative_columns]


final_confounds = confound_vars
final_confounds = confound_vars + derivative_columns 

P_N = [[100,7]]
functional_datasets_w= LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing, final_confounds)
         
       

#%%
root = 'D:\\fmri_preproc_fmap' 
atlas_dict_p = 'C:\\Users\\danie\\phd\\pharmo-fmri\\scripts\\FunctionalFeatures\\LUTDict'

preprocessing = [ 'AROMA_bold_no_mask']

P_N = [[100, 7],[200, 7],[100, 17],[500, 17],[1000, 7],[1000, 17]]

functional_datasets = LoopFmriprepOutput.loop_fmriprep_output(root, atlas_dict_p, P_N, preprocessing)
         
#%%
import pickle
with open('functional_datasets_h.pkl', 'rb') as file:
    functional_datasets_h  = pickle.load(file)


#%%
feature_list = ['seg','integ','fcs_var','dfcs_mean_segs', 'dfcs_mean_integs','fcd_var','alff','falff']        
#%%
results = pd.read_csv('C:\\Users\\danie\\phd\\pharmo-fmri\\results 30.0\\Results k=2.csv')


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
    
    
    c0_seg = seg_class[seg_class['Clusters'] == 0]; c1_seg = seg_class[seg_class['Clusters'] == 1]
    c0_seg = c0_seg.drop(columns=['Clusters']); c1_seg = c1_seg.drop(columns=['Clusters'])
    
    significance_seg= []
    for feature in c0_seg.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature], c1_seg[feature])
        significance_seg.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_integ = integ_class[integ_class['Clusters'] == 0]; c1_integ = integ_class[integ_class['Clusters'] == 1]
    c0_integ = c0_integ.drop(columns=['Clusters']); c1_integ = c1_integ.drop(columns=['Clusters'])
    
    significance_integ= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_integ[feature], c1_integ[feature])
        significance_integ.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    
    
    significance_ratio= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature]/c0_integ[feature], c1_seg[feature]/c1_integ[feature])
        significance_ratio.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    



    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.10)
    Q3 = SegInteg['Value'].quantile(0.90)
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
    
    for i, feature in enumerate(c0_seg.columns):
        pval = significance_seg[i]["p-value"]
        y_position = features_seg['Value'].max() + 0.1
        axes[0].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[0].hlines(y=y_position - 0.02, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)


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
    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_integ[i]["p-value"]
        y_position = features_integ['Value'].max() + 0.05
        axes[1].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[1].hlines(y=y_position - 0.01, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
        

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
    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_ratio[i]["p-value"]
        y_position = filtered_data['Value'].max() - 3
        axes[2].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[2].hlines(y=y_position - 0.1, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
        

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
    axes[2].set_title('Ratio Seg/Integ')


    axes[0].set_ylim(bottom=0, top=features_seg['Value'].max() + 0.2)  
    axes[1].set_ylim(bottom=0, top=features_integ['Value'].max() + 0.08)  

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc} CSF, GS, WM derivatives and quadratic terms")
    plt.tight_layout()  
    plt.show()

#%% alf, falf
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
    
    c0_seg = seg_class[seg_class['Clusters'] == 0]; c1_seg = seg_class[seg_class['Clusters'] == 1]
    c0_seg = c0_seg.drop(columns=['Clusters']); c1_seg = c1_seg.drop(columns=['Clusters'])
    
    significance_seg= []
    for feature in c0_seg.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature], c1_seg[feature])
        significance_seg.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_integ = integ_class[integ_class['Clusters'] == 0]; c1_integ = integ_class[integ_class['Clusters'] == 1]
    c0_integ = c0_integ.drop(columns=['Clusters']); c1_integ = c1_integ.drop(columns=['Clusters'])
    
    significance_integ= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_integ[feature], c1_integ[feature])
        significance_integ.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    

    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
    
    significance_ratio= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature]/c0_integ[feature], c1_seg[feature]/c1_integ[feature])
        significance_ratio.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    

        
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
       
    for i, feature in enumerate(c0_seg.columns):
        pval = significance_seg[i]["p-value"]
        y_position = features_seg['Value'].max() + 0.005
        axes[0].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[0].hlines(y=y_position - 0.002, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)

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

    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_integ[i]["p-value"]
        y_position = features_integ['Value'].max() + 0.7
        axes[1].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[1].hlines(y=y_position - 0.1, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
        
    axes[0].set_ylim(bottom=0.035, top=features_seg['Value'].max() + 0.0099)  
    axes[1].set_ylim(bottom=4, top=features_integ['Value'].max() + 2)  

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc} CSF, GS, WM, regressed out derivatives and quadratic terms")
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

        seg = pd.concat([seg,dataset['data'][key]['dfcs_mean_segs'].T],axis =1)
        integ = pd.concat([integ,dataset['data'][key]['dfcs_mean_integs']],axis = 1)
        networks = dataset['data'][key]['fc'].index
       
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
    
    
    c0_seg = seg_class[seg_class['Clusters'] == 0]; c1_seg = seg_class[seg_class['Clusters'] == 1]
    c0_seg = c0_seg.drop(columns=['Clusters']); c1_seg = c1_seg.drop(columns=['Clusters'])
    
    significance_seg= []
    for feature in c0_seg.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature], c1_seg[feature])
        significance_seg.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_integ = integ_class[integ_class['Clusters'] == 0]; c1_integ = integ_class[integ_class['Clusters'] == 1]
    c0_integ = c0_integ.drop(columns=['Clusters']); c1_integ = c1_integ.drop(columns=['Clusters'])
    
    significance_integ= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_integ[feature], c1_integ[feature])
        significance_integ.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    


    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.10)
    Q3 = SegInteg['Value'].quantile(0.90)
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

    axes[0].set_title('Mean dFC Segreation by Cluster')
    
    for i, feature in enumerate(c0_seg.columns):
        pval = significance_seg[i]["p-value"]
        y_position = features_seg['Value'].max() + 0.1
        axes[0].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[0].hlines(y=y_position - 0.02, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)


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

    axes[1].set_title('Mean dFC Integration by Cluster')
    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_integ[i]["p-value"]
        y_position = features_integ['Value'].max() + 0.05
        axes[1].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[1].hlines(y=y_position - 0.01, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
        
 

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
    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_ratio[i]["p-value"]
        y_position = filtered_data['Value'].max() - 3
        axes[2].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[2].hlines(y=y_position - 0.1, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)

    _ = [label.set_rotation(30) or label.set_ha('right') for label in axes[1].get_xticklabels()]
    axes[2].set_title('Ratio Mean dFC Seg/Integ')


    axes[0].set_ylim(bottom=0, top=features_seg['Value'].max() + 0.2)  
    axes[1].set_ylim(bottom=0, top=features_integ['Value'].max() + 0.08)  

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc} CSF, GS, WM regressed out derivatives and quadratic terms")
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

        seg = pd.concat([seg,dataset['data'][key]['fcd_means'].T],axis =1)
        integ = pd.concat([integ,dataset['data'][key]['fcd_vars']],axis = 1)
        networks = dataset['data'][key]['fc'].index
       
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
    
    
    c0_seg = seg_class[seg_class['Clusters'] == 0]; c1_seg = seg_class[seg_class['Clusters'] == 1]
    c0_seg = c0_seg.drop(columns=['Clusters']); c1_seg = c1_seg.drop(columns=['Clusters'])
    
    significance_seg= []
    for feature in c0_seg.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature], c1_seg[feature])
        significance_seg.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_integ = integ_class[integ_class['Clusters'] == 0]; c1_integ = integ_class[integ_class['Clusters'] == 1]
    c0_integ = c0_integ.drop(columns=['Clusters']); c1_integ = c1_integ.drop(columns=['Clusters'])
    
    significance_integ= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_integ[feature], c1_integ[feature])
        significance_integ.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    


    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.10)
    Q3 = SegInteg['Value'].quantile(0.90)
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

    axes[0].set_title('FCD Mean')
    
    for i, feature in enumerate(c0_seg.columns):
        pval = significance_seg[i]["p-value"]
        y_position = features_seg['Value'].max() + 0.1
        axes[0].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[0].hlines(y=y_position - 0.02, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)


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

    axes[1].set_title('FCD Var')
    
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_integ[i]["p-value"]
        y_position = features_integ['Value'].max() + 0.05
        axes[1].annotate(f'p={pval:.3f}', xy=(i, y_position), ha='center', color='black', fontsize=10)
        axes[1].hlines(y=y_position - 0.01, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
        

    axes[0].set_ylim(bottom=0, top=features_seg['Value'].max() + 0.2)  
    axes[1].set_ylim(bottom=0, top=features_integ['Value'].max() + 0.08)  

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.suptitle(f"{N_n} Networks {N_p} Parcelations {preproc} CSF, GS, WM regressed out derivatives and quadratic terms")
    plt.tight_layout()  
    plt.show()

#%%
data = functional_datasets[0].get('data')
fcs=[]
for key, features in data.items():
    fcs.append(features.get("fc"))
#%%
labels = fcs[i].columns
#%%
unique = []  # List to store new unique labels
label_count = {}  # Dictionary to count occurrences
for label in labels:
    if label in label_count:
        label_count[label] += 1  # Increment the count for the label
    else:
        label_count[label] = 1  # Initialize the count

    # Append the label with its count as a suffix
    unique.append(f"{label}_{label_count[label]}")

#%%
fcs = np.array(fcs)
fcs = np.mean(fcs, axis = 0)
mean_fc = pd.DataFrame(fcs, index = unique, columns = unique)
#%%
sorted_columns = sorted(mean_fc.columns)
sorted_data = mean_fc[sorted_columns]
sorted_data.sort_index(inplace = True)
#%%
revert_columns = {new_name: '_'.join(new_name.split('_')[0:3]) for new_name in sorted_data.columns}
revert_index = {index_name: '_'.join(index_name.split('_')[0:3]) for index_name in sorted_data.index}

sorted_data.rename(columns=revert_columns, inplace=True)
sorted_data.rename(index=revert_index, inplace=True)

#%%
plt.figure()
sns.heatmap(sorted_data,  cmap = 'bwr')
plt.xticks(rotation=45, ha='right')
#plt.savefig("5").png

#%%
#%%
import ptitprince as pt 
for dataset in  functional_datasets:
    
   
    nident = []
    seg = pd.DataFrame()
    integ = pd.DataFrame() 
    metastability = []
    for i, key in enumerate(functional_datasets["data"].keys()):
        
        nident.append(key)
        
        
        metastability.append(functional_datasets['data'][key]['metastability'])
        seg = pd.concat([seg,functional_datasets['data'][key]['fcd_means'].T],axis =1)
        integ = pd.concat([integ,functional_datasets['data'][key]['fcd_vars']],axis = 1)
        networks = functional_datasets['data'][key]['fc'].index
       
    seg = seg.T
    integ = integ.T
    metastability = pd.DataFrame(np.array(metastability))    
    metastability.reset_index(drop = True, inplace = True); metastability.set_index(np.array(nident), inplace = True)
    seg.reset_index(drop = True, inplace = True); seg.set_index(np.array(nident), inplace = True)
    integ.reset_index(drop = True, inplace = True); integ.set_index(np.array(nident), inplace = True)
    
    print (seg.head())
    print("-"*65)
    print(integ.head())
    print("-"*65)

    seg_class = seg.merge(results[['Clusters']], left_index=True, right_index=True, how = 'inner')
    integ_class = integ.merge(results[['Clusters']], left_index=True, right_index = True, how ='inner')
    meta_class = metastability.merge(results[['Clusters']], left_index=True, right_index = True, how ='inner')
    
    c0_seg = seg_class[seg_class['Clusters'] == 0]; c1_seg = seg_class[seg_class['Clusters'] == 1]
    c0_seg = c0_seg.drop(columns=['Clusters']); c1_seg = c1_seg.drop(columns=['Clusters'])
    
    significance_seg= []
    for feature in c0_seg.columns:
        u_stat, p_value = st.mannwhitneyu(c0_seg[feature], c1_seg[feature])
        significance_seg.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_meta = meta_class[meta_class['Clusters'] == 0];  c1_meta = meta_class[meta_class['Clusters'] == 1]
    c0_meta = c0_meta.drop(columns=['Clusters']); c1_meta = c1_meta.drop(columns=['Clusters'])

    significance_meta= []
    for feature in c0_meta.columns:
        u_stat, p_value = st.mannwhitneyu(c0_meta[feature], c1_meta[feature])
        significance_meta.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})
    
    c0_integ = integ_class[integ_class['Clusters'] == 0]; c1_integ = integ_class[integ_class['Clusters'] == 1]
    c0_integ = c0_integ.drop(columns=['Clusters']); c1_integ = c1_integ.drop(columns=['Clusters'])
    
    significance_integ= []
    for feature in c0_integ.columns:
        u_stat, p_value = st.mannwhitneyu(c0_integ[feature], c1_integ[feature])
        significance_integ.append({"Feature": feature, "u-statistic": u_stat, "p-value": p_value})    


    features_seg = pd.melt(seg_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_integ = pd.melt(integ_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')
    features_meta = pd.melt(meta_class, id_vars='Clusters', var_name = 'Feature', value_name = 'Value')

    ratio = features_seg['Value']/features_integ['Value']
        
    SegInteg = features_seg.copy() 
    SegInteg['Value'] = ratio
    
    Q1 = SegInteg['Value'].quantile(0.10)
    Q3 = SegInteg['Value'].quantile(0.90)
    IQR = Q3 - Q1

    lower_bound = Q1 - 0.5 * IQR
    upper_bound = Q3 + 0.5 * IQR

    filtered_data = SegInteg[(SegInteg['Value'] >= lower_bound) & (SegInteg['Value'] <= upper_bound)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))
    colors = ['#FF0101', '#0E0EFF']

    def get_star_annotation(pval):
        """Returns the star annotation based on the p-value."""
        if pval <= 0.001:
            return '***'
        elif pval <= 0.01:
            return '**'
        elif pval <= 0.05:
            return '*'
        else:
            return ''

# Segregated FC by Cluster
    pt.RainCloud(
        x='Feature', y='Value', hue='Clusters', data=features_seg,
        palette=colors, width_viol=0.6, ax=axes[0], dodge=True, orient='v',
        alpha=.65, move=.0
        )
    axes[0].set_title('FALFF')

# Annotate stars for significance for Segregated FC
    for i, feature in enumerate(c0_seg.columns):
        pval = significance_seg[i]["p-value"]
        star_annotation = get_star_annotation(pval)
        y_position = features_seg['Value'].max() + 0.1
        if star_annotation:  # Only annotate if there is a star (significant)
            axes[0].annotate(star_annotation, xy=(i, y_position), ha='center', color='black', fontsize=12)
        axes[0].hlines(y=y_position - 0.02, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)

# Integrated FC by Cluster
    pt.RainCloud(
        x='Feature', y='Value', hue='Clusters', data=features_integ,
        palette=colors, width_viol=0.6, ax=axes[1], dodge=True, orient='v',
        alpha=.65, move=.0
        )
    axes[1].set_title('ALFF')

# Annotate stars for significance for Integrated FC
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_integ[i]["p-value"]
        star_annotation = get_star_annotation(pval)
        y_position = features_integ['Value'].max() + 0.05
        if star_annotation:
            axes[1].annotate(star_annotation, xy=(i, y_position), ha='center', color='black', fontsize=12)
        axes[1].hlines(y=y_position - 0.01, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)

# Ratio Seg/Integ
    pt.RainCloud(
        x='Feature', y='Value', hue='Clusters', data=filtered_data,
        palette=colors, width_viol=0.6, ax=axes[2], dodge=True, orient='v',
        alpha=.65, move=.0
        )
    axes[2].set_title('Ratio Seg/Integ')

# Annotate stars for significance for Ratio
    for i, feature in enumerate(c0_integ.columns):
        pval = significance_ratio[i]["p-value"]
        star_annotation = get_star_annotation(pval)
        y_position = filtered_data['Value'].max() - 3
        if star_annotation:
            axes[2].annotate(star_annotation, xy=(i, y_position), ha='center', color='black', fontsize=12)
        axes[2].hlines(y=y_position - 0.1, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)

# Rotate x-axis labels and remove legends
    for ax in axes:
        _ = [label.set_rotation(30) or label.set_ha('right') for label in ax.get_xticklabels()]
        ax.get_legend().remove()

# Set axis limits and titles
    axes[0].set_ylim(bottom=0, top=features_seg['Value'].max() + 0.2)
    axes[1].set_ylim(bottom=0, top=features_integ['Value'].max() + 0.08)
    fig.suptitle("7 Networks 100 Parcelations MNI CSF, GS, WM derivatives and quadratic terms AromaSmooth")
    plt.tight_layout()
    plt.show()

#%%
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# Segregated FC by Cluster
    pt.RainCloud(
        x='Feature', y='Value', hue='Clusters', data=features_meta,
        palette=colors, width_viol=0.6, ax=axes[0], dodge=True, orient='v',
        alpha=.65, move=.0
        )
    axes[0].set_title('FALFF')

# Annotate stars for significance for Segregated FC
    for i, feature in enumerate(c0_meta.columns):
        pval = significance_meta[i]["p-value"]
        star_annotation = get_star_annotation(pval)
        y_position = features_meta['Value'].max() + 0.1
        if star_annotation:  # Only annotate if there is a star (significant)
            axes[0].annotate(star_annotation, xy=(i, y_position), ha='center', color='black', fontsize=12)
        axes[0].hlines(y=y_position - 0.02, xmin=i - 0.2, xmax=i + 0.2, color='black', linewidth=1)
