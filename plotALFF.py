# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:42:48 2024

@author: danie
"""

import numpy as np
import nibabel as nib
from scipy.signal import butter, filtfilt
from nilearn import image, plotting

# Function to apply a Butterworth bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Load BOLD fMRI data
bold_img = nib.load('path_to_your_bold_file.nii.gz')
bold_data = bold_img.get_fdata()

# Get dimensions
n_timepoints = bold_data.shape[3]
n_voxels = np.prod(bold_data.shape[:3])

# Prepare an empty array for ALFF
alff_data = np.zeros(bold_data.shape[:3])

# Bandpass filter parameters
lowcut = 0.01  # Hz
highcut = 0.08  # Hz
fs = 1.0 / 3.6  # Replace TR with your repetition time (in seconds)

# Compute ALFF for each voxel
for x in range(bold_data.shape[0]):
    for y in range(bold_data.shape[1]):
        for z in range(bold_data.shape[2]):
            voxel_time_series = bold_data[x, y, z, :]
            filtered_series = bandpass_filter(voxel_time_series, lowcut, highcut, fs)
            alff_data[x, y, z] = np.mean(np.abs(filtered_series))

# Save ALFF as a NIfTI image
alff_img = nib.Nifti1Image(alff_data, bold_img.affine)
nib.save(alff_img, 'alff_output.nii.gz')

# Load and visualize ALFF
alff_img = image.load_img('alff_output.nii.gz')
display = plotting.plot_stat_map(alff_img,
                                  threshold=0.1,  # Adjust threshold as needed
                                  title='ALFF',
                                  cut_coords=(-24, -6, 30),  # Example coordinates
                                  display_mode='ortho',
                                  colorbar=True)

# Show the plot
plotting.show()
