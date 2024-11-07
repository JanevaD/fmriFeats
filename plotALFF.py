import numpy as np
import nibabel as nib
import scipy as sp
from scipy.signal import butter, filtfilt
from nilearn import image, plotting, signal
import pandas as pd

# Function to apply a Butterworth bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Function to compute ALFF and fALFF
def get_alff_falff(time_series, tr):
    """
    Calculate ALFF (Amplitude of Low Frequency Fluctuations) and fALFF (fractional ALFF)
    for each regional time series.
    """
    alffs = []
    falffs = []

    for i in range(len(time_series)):
        # Detrend the time series for the i-th region
        detrended = sp.signal.detrend(time_series.iloc[:, i])

        # Compute the power spectral density (PSD) using the Welch method
        f, Pxx = sp.signal.welch(detrended, fs=1/tr, nperseg=64)

        # Select the low frequency range (0.01 Hz to 0.08 Hz for ALFF)
        low_freq_indices = np.where((f >= 0.01) & (f <= 0.08))

        # Compute ALFF: the sum of power in the low-frequency band
        alff = np.sqrt(np.sum(Pxx[low_freq_indices]))
        alffs.append(alff)

        # Compute fALFF: the ratio of ALFF to total power
        falff = alff / np.sum(Pxx)
        falffs.append(falff)

    return alffs, falffs

# Load BOLD fMRI data
confound_vars = ['global_signal', 'csf', 'white_matter']
derivative_columns = ['{}_derivative1'.format(c) for c in confound_vars]
confound_vars_power2 = ['{}_power2'.format(c) for c in confound_vars]
derivative_power2 =  ['{}_power2'.format(c) for c in derivative_columns]
final_confounds = confound_vars + derivative_columns + confound_vars_power2 + derivative_power2

fmri_path = 'D:/fmri_preproc_fmap/sub-007/ses-V0/func/sub-007_ses-V0_task-rest_run-01_space-T1w_desc-AROMA_bold_no_mask.nii.gz'
confounds_path = 'D:/fmri_preproc_fmap/sub-007/ses-V0/func/sub-007_ses-V0_task-rest_run-01_desc-confounds_timeseries.tsv'

# Load the fMRI image, excluding the first slice
fmri = image.load_img(fmri_path)
fmri = fmri.slicer[:,:,:,1:]  # Exclude the first slice

# Load confounds timeseries
confounds_timeseries = pd.read_csv(confounds_path, sep='\t')
confounds = confounds_timeseries[final_confounds][1::]

# Bandpass filter parameters
lowcut = 0.01  # Hz
highcut = 0.08  # Hz
fs = 1.0 / 3.6  # Replace TR with your repetition time (in seconds)

# Regress out confounds using nilearn's image.clean_img
confounds_clean = confounds.values  # Convert to numpy array

# Clean the BOLD data using nilearn's image.clean_img
cleaned_bold_img = image.clean_img(fmri, confounds=confounds_clean, detrend=True, standardize=True)

# Get cleaned BOLD data from the cleaned image
cleaned_bold_data = cleaned_bold_img.get_fdata()

# Prepare an empty array for ALFF and fALFF
alff_data = np.zeros(cleaned_bold_data.shape[:3])
falff_data = np.zeros(cleaned_bold_data.shape[:3])

# Compute ALFF and fALFF for each voxel on the cleaned data
for x in range(1, cleaned_bold_data.shape[0]):  # Start from the second slice
    for y in range(cleaned_bold_data.shape[1]):
        for z in range(cleaned_bold_data.shape[2]):
            voxel_time_series = cleaned_bold_data[x, y, z, :]
            # Convert to DataFrame for easier handling in the ALFF/fALFF function
            voxel_time_series_df = pd.DataFrame(voxel_time_series).T
            # Compute ALFF and fALFF using the get_alff_falff function
            alff, falff = get_alff_falff(voxel_time_series_df, fs)
            alff_data[x, y, z] = alff[0]  # Get ALFF for this voxel
            falff_data[x, y, z] = falff[0]  # Get fALFF for this voxel

# Save ALFF and fALFF as NIfTI images
alff_img = nib.Nifti1Image(alff_data, fmri.affine)
nib.save(alff_img, 'alff_output_no_first_slice.nii.gz')

falff_img = nib.Nifti1Image(falff_data, fmri.affine)
nib.save(falff_img, 'falff_output_no_first_slice.nii.gz')

# Load and visualize ALFF
alff_img = image.load_img('alff_output_no_first_slice.nii.gz')
display = plotting.plot_stat_map(alff_img,
                                  threshold=0.1,  # Adjust threshold as needed
                                  title='ALFF Excluding First Slice',
                                  cut_coords=(-24, -6, 30),  # Example coordinates
                                  display_mode='ortho',
                                  colorbar=True)

# Load and visualize fALFF
falff_img = image.load_img('falff_output_no_first_slice.nii.gz')
display = plotting.plot_stat_map(falff_img,
                                  threshold=0.1,  # Adjust threshold as needed
                                  title='fALFF Excluding First Slice',
                                  cut_coords=(-24, -6, 30),  # Example coordinates
                                  display_mode='ortho',
                                  colorbar=True)

# Show the plots
plotting.show()
