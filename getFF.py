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
    names = ['_'.join(name.split('_')[1:3]) for name in fc.columns]
    fc.columns = names
    fc.index = names
    
    return fc

def get_fc_seg_integ(time_series):
    """
    Function to calculate FC Segregation and Integration 
    
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
    names = ['_'.join(name.split('_')[1:3]) for name in fc.columns]
    fc.columns = names
    fc.index = names
    
    seg_fcs =[]
    integ_fcs=[]
    networks = set([name.split('_')[1] for name in names]) 
 
    for network in networks:        
        columns = [col for col in fc.columns if f'_{network}' in col]
        num_columns = len(columns)
        
        seg_fc  = fc.copy()
        seg_fc = seg_fc.loc[columns,columns].sum().sum()
        seg_fcs.append(seg_fc/num_columns)
        print(seg_fcs)
        integ_fc = fc.copy()
        integ_fc.loc[columns,columns]=0
        
        integ_fc = integ_fc.loc[columns,::].sum().sum()
        integ_fcs.append(integ_fc/(np.array(fc.shape[1])-num_columns))
        print(integ_fcs)

    seg_fcs = pd.Series(seg_fcs,index = list(networks))
    integ_fcs = pd.Series(integ_fcs, index = list(networks))
        
    return seg_fcs, integ_fcs   


def get_dfc_feats (time_series,L=15, S=2):
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
    
    fc_stream =[]
    M = len(time_series)
    for i in range (0, M-S, S):
        correlation_measure = ConnectivityMeasure(kind='correlation')
        dfc = correlation_measure.fit_transform([time_series[i:i+L].values])[0]
       # dfc = np.tril(dfc, k=-1).flatten()
        #dfc = dfc[dfc!=0] 
        fc_stream.append(dfc)        
        
    fc_stream= np.stack(fc_stream)
    
    #fc_stream = dim(fc_stream, int(np.ceil((len(time_series)-L)/S)+1 ))
    fcs_var = np.var(fc_stream, axis = 0)
    fcd = np.mean(fc_stream, axis = 0)
    
    return fc_stream, fcs_var, fcd

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
        detrended = sp.signal.detrend(time_series.iloc[i])
        f, Pxx = sp.signal.welch(detrended, fs=1/tr, nperseg = 64)
   
        low_freq_indices = np.where((f >= 0.01) & (f <= 0.08))
        alff = np.sqrt(np.sum(Pxx[low_freq_indices]))
        alffs.append(alff)
        falff = alff/np.sum(Pxx)
        falffs.append(falff)
        
    return alffs, falffs