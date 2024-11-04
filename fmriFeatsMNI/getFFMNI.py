# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:09:38 2024

@author: danie
"""

from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import numpy as np
import scipy as sp

def get_fc(time_series):
    """
    Function to calculate the functional connectivity i.e correlation between regional BOLD signals
    
    :param time_series: regional BOLD signals
    :type time_series: DataFrame
    :return: Functional Connectivity
    :rtype: DataFrame

    """
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series.values])[0]
    
    fc = pd.DataFrame(correlation_matrix, columns= time_series.columns, index = time_series.columns)
    names = ['_'.join(name.split('_')[2:3]) for name in fc.columns]
    fc.columns = names
    fc.index = names
    
    return fc

def get_fc_seg_integ(time_series):
    """
    Function to calculate FC Segregation and Integration of Whole Brain Networks
    
    :param fc: functional connectivity
    :type fc: DataFrame
    :param networks: network labels
    :type networks: list
    :return: functional connectivity segregation and integration
    :rtype: lists

    """
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series.values])[0]
    
    fc = pd.DataFrame(correlation_matrix, columns= time_series.columns, index = time_series.columns)
    names = ['_'.join(name.split('_')[2:3]) for name in fc.columns]
    fc.columns = names
    fc.index = names
    
    
    seg_fcs =[]
    integ_fcs=[]
    networks = set(names) 
    
    for network in networks:        
        
        num_columns = int(len([col for col in fc.columns if col == network]))
        
        seg_fc  = fc.copy()
        seg_fc = seg_fc.loc[network,network].sum().sum()
        seg_fcs.append(seg_fc/(num_columns*num_columns))
 
        integ_fc = fc.copy()
        integ_fc.loc[network,network]=0
        
        integ_fc = integ_fc.loc[network,::].sum().sum()
        i_n = (np.array(fc.shape[1])-num_columns)
        integ_fcs.append(integ_fc/(i_n*i_n))
        

    seg_fcs = pd.Series(seg_fcs,index = list(networks))
    integ_fcs = pd.Series(integ_fcs, index = list(networks))
        
    return seg_fcs, integ_fcs   


def get_dfc_feats(time_series,L=15, S=2):
    """    
    :param time_series: Regional BOLD 
    :type time_series: DataFrame
    :param L: window length 
    :type L: int
    :param S: stepsize
    :type S: int 
    :return: functional connectivity stream, stream variance and fcd
    :rtype: TYPE
    """
 
    names = ['_'.join(name.split('_')[2:3]) for name in time_series.columns]
    networks = set(names)
    
    fc_stream =[]
    dfcs = []
    M = len(time_series)
    for i in range (0, M-S, S):
        correlation_measure = ConnectivityMeasure(kind='correlation')
        dfc = correlation_measure.fit_transform([time_series[i:i+L].values])[0]
        dfcs.append(dfc)
        dfc_t = np.tril(dfc, k=-1).flatten()
        fc_stream.append(dfc_t)    
    
    dfcs = np.array(dfcs)   
    dfcs_mean = np.mean(dfcs,axis = 0)

    dfcs_mean = pd.DataFrame(dfcs_mean, index=names, columns = names)

    dfcs_mean_segs=[]
    dfcs_mean_integs=[]
    
    for network in networks:        
        
        num_columns = int(len([col for col in names if col == network]))
        
        dfcs_mean_seg  = dfcs_mean.copy()
        dfcs_mean_seg = dfcs_mean_seg.loc[network,network].sum().sum()
        dfcs_mean_segs.append(dfcs_mean_seg/(num_columns*num_columns))
 
        dfcs_mean_integ = dfcs_mean.copy()
        dfcs_mean_integ.loc[network,network]=0
        
        dfcs_mean_integ = dfcs_mean_integ.loc[network,::].sum().sum()
        i_n = (np.array(dfcs_mean.shape[1])-num_columns)
        dfcs_mean_integs.append(dfcs_mean_integ /(i_n*i_n))

    dfcs_mean_segs = pd.Series(dfcs_mean_segs,index = list(networks))
    dfcs_mean_integs = pd.Series(dfcs_mean_integs, index = list(networks))
    
    fc_stream= np.stack(fc_stream); fcs_var = np.var(fc_stream)
    fcd = np.corrcoef(fc_stream)
    fcd_var = np.var(np.triu(fcd, k=L-S).flatten())
    fcd_mean = np.mean(np.triu(fcd, k=L-S).flatten())
    
    return fcs_var, dfcs_mean_segs, dfcs_mean_integs, fcd_var, fcd_mean


def get_fluidity_feats(time_series,L=15, S=2):
    """    
    :param time_series: Regional BOLD 
    :type time_series: DataFrame
    :param L: window length 
    :type L: int
    :param S: stepsize
    :type S: int 
    :return: functional connectivity stream, stream variance and fcd
    :rtype: TYPE
    """
    names = ['_'.join(name.split('_')[2:3]) for name in time_series.columns]
    networks = set(names)
    time_series.columns = names
    fc_stream =[]
    fcd_vars = []
    fcd_means = []
    M = len(time_series)
    
    for network in networks:
        fc_stream =[]
        print (network)
        for i in range (0, M-S, S):
            correlation_measure = ConnectivityMeasure(kind='correlation')
            dfc = correlation_measure.fit_transform([np.array(time_series.loc[:,network].iloc[i:i+L].values)])[0]
            dfc_t = np.tril(dfc, k=M-S).flatten()
            fc_stream.append(dfc_t)    
        fcd = np.corrcoef(fc_stream)
        fcd_vars.append(np.var(np.triu(fcd, k=L-S).flatten()))
        fcd_means.append(np.mean(np.triu(fcd, k=L-S).flatten()))
        
    fcd_vars = pd.Series(fcd_vars,index = list(networks))
    fcd_means = pd.Series(fcd_means, index = list(networks))
    
    return fcd_vars, fcd_means


def get_falff(time_series, tr):
    
    """
    :param time_series: regional bold timeseries
    :type time_series: df
    :param tr: time repetition
    :type tr: int
    :return: functional amplitude of low frequency fluctuations
    :rtype: TYPE

    """
    alffs = []
    falffs = []
    
    for i in range (time_series.shape[1]): 
        detrended = sp.signal.detrend(time_series.iloc[:,i])
        f, Pxx = sp.signal.welch(detrended, fs=1/tr, nperseg = 64)
   
        low_freq_indices = np.where((f >= 0.01) & (f <= 0.08))
        alff = np.sqrt(np.sum(Pxx[low_freq_indices]))
        alffs.append(alff)
        falff = alff/np.sum(Pxx)
        falffs.append(falff)
        
    return alffs, falffs