# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:09:38 2024

@author: danie
"""

from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
from scipy.linalg import eigh

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
    names = ['_'.join(name.split('_')[1:3]) for name in fc.columns]
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
 
    names = ['_'.join(name.split('_')[1:3]) for name in time_series.columns]
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
    names = ['_'.join(name.split('_')[1:3]) for name in time_series.columns]
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



def calculate_phase_locking(time_series):
    """
    

    Parameters
    ----------
    time_series : pandas dataframe con
        DESCRIPTION.

    Returns
    -------
    phase_locking_matrix : TYPE
        DESCRIPTION.

    """
    """
    Calculate instantaneous phase-locking values using the Hilbert transform on parcellated BOLD time series.
    """
    analytic_signal = hilbert(time_series.values, axis=0)
    phase_data = np.angle(analytic_signal)
    
    n_regions = phase_data.shape[1]
    phase_locking_matrix = np.zeros((time_series.shape[0], n_regions, n_regions))
    
    for t in range(time_series.shape[0]):
        for i in range(n_regions):
            for j in range(n_regions):
                phase_locking_matrix[t, i, j] = np.cos(phase_data[t, i] - phase_data[t, j])
    
    return phase_locking_matrix

def extract_leida_modes(phase_locking_matrix, n_clusters=5):
    """
    Extract LEiDA modes using k-means clustering on the leading eigenvector of phase-locking matrices.
    """
    n_timepoints, n_regions, _ = phase_locking_matrix.shape
    leida_vectors = np.zeros((n_timepoints, n_regions))
    
    for t in range(n_timepoints):
        # Calculate the leading eigenvector of the phase-locking matrix at time t
        eigvals, eigvecs = eigh(phase_locking_matrix[t])
        leading_eigvec = eigvecs[:, -1]  # leading eigenvector
        leida_vectors[t] = leading_eigvec
    
    # Standardize leading eigenvectors before clustering
    scaler = StandardScaler()
    leida_vectors = scaler.fit_transform(leida_vectors)
    
    # Perform k-means clustering on the leading eigenvectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(leida_vectors)
    
    return labels, kmeans.cluster_centers_

def calculate_metastability_from_modes(labels, phase_locking_matrix):
    """
    Calculate metastability as the variance of phase-locking within each LEiDA mode.
    """
    metastability_values = []
    unique_modes = np.unique(labels)
    
    for mode in unique_modes:
        mode_indices = np.where(labels == mode)[0]
        mode_phase_locking = phase_locking_matrix[mode_indices]
        
        variances = np.var(mode_phase_locking, axis=0).mean()
        metastability_values.append(variances)
    
    ms= np.mean(metastability_values)
    return ms

def get_metastability(time_series):
    
    phase_locking_matrix = calculate_phase_locking(time_series)
    labels, mode_centers = extract_leida_modes(phase_locking_matrix)
    metastability = calculate_metastability_from_modes(labels, phase_locking_matrix)
    
    return metastability

def get_metastability_standard(time_series):
    """
    Calculate metastability as the mean variance of instantaneous phase-locking (VAR).
    
    :param time_series: regional BOLD signals
    :type time_series: DataFrame
    :return: Metastability (mean variance of instantaneous phase-locking)
    :rtype: float
    """
    # Calculate instantaneous phase using Hilbert transform
    phase_data = np.angle(np.apply_along_axis(lambda x: np.fft.ifft(np.fft.fft(x)), 0, time_series))
    
    # Calculate instantaneous phase-locking for each pair of regions
    n_regions = phase_data.shape[1]
    phase_locking_values = np.zeros((time_series.shape[0], int(n_regions * (n_regions - 1) / 2)))
    idx = 0
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            phase_locking_values[:, idx] = np.cos(phase_data[:, i] - phase_data[:, j])
            idx += 1
    
    # Calculate metastability as the mean variance of instantaneous phase-locking
    ms_st = np.mean(np.var(phase_locking_values, axis=0))
    
    return ms_st

def calculate_rsfa(time_series):
    """
    Calculate RSFA (Resting-State Fluctuation Amplitude) as the standard deviation of the time series for each region.
    """
    rsfa = time_series.std(axis=0)
    return rsfa

def calculate_dvars(time_series):
    """
    Calculate DVARS (D temporal derivative of timecourses VARiance) as the root mean square of temporal derivatives.
    """
    temporal_derivative = np.diff(time_series, axis=0)
    dvars = np.sqrt((temporal_derivative**2).mean(axis=0))
    return dvars

def calculate_reho(time_series, k=27):
    """
    Calculate ReHo (Regional Homogeneity) as Kendall's coefficient of concordance for neighboring voxels.
    """
    # Assuming 'time_series' is already spatially smoothed; if not, smoothing is recommended for ReHo
    # Implementing ReHo over parcels using Kendall's coefficient
    
    # Here we'll calculate ReHo as an average correlation across neighboring parcels.
    reho = []
    for i in range(time_series.shape[1]):
        # Select k-nearest neighbors based on time series similarity for each parcel
        correlations = np.corrcoef(time_series.iloc[:, i], time_series)
        sorted_neighbors = np.argsort(-np.abs(correlations))[:k+1]  # top k neighbors
        reho_val = np.mean(correlations[sorted_neighbors])
        reho.append(reho_val)
    reho = pd.Series(reho, index=time_series.columns)
    return reho

def calculate_tv(time_series):
    """
    Calculate Temporal Variability (TV) as the standard deviation of ALFF over time.
    """
    # Calculate instantaneous amplitude (ALFF) using the Hilbert transform
    analytic_signal = hilbert(time_series, axis=0)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Temporal variability is the standard deviation of this amplitude envelope over time
    tv = amplitude_envelope.std(axis=0)
    return tv

def calculate_ih(time_series, hemisphere_pairs):
    """
    Calculate Inter-hemispheric Homotopic Connectivity (IH) for given pairs of regions in each hemisphere.
    
    :param hemisphere_pairs: List of tuples, each containing a pair of columns representing homotopic regions
    """
    ih_values = []
    for left, right in hemisphere_pairs:
        ih_corr = np.corrcoef(time_series[left], time_series[right])[0, 1]
        ih_values.append(ih_corr)
    
    ih = pd.Series(ih_values, index=[f"{pair[0]}-{pair[1]}" for pair in hemisphere_pairs])
    return ih

