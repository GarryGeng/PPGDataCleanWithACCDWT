import numpy as np
# from utils import *
import matplotlib.pyplot as plt
import torch
from scipy.signal import find_peaks
from model import Model
import pywt  # Added import for pywt
import neurokit2 as nk

model = Model()

# Load data from PPGData.txt with a comma delimiter, integer data type, and filling missing values
data = []
raw_signals = []  # Store raw data (before wavelet transform) for SQI
orig_lengths = []  # Store original lengths
max_columns = 1920

with open('PPGData.txt', 'r') as file:
# with open('PPGDataFull.txt', 'r') as file:
    wavelet = pywt.Wavelet('coif1')  # Define wavelet
    for line in file:
        row = list(map(int, line.strip().split(',')))
        raw_signals.append(row)  # Keep a copy of the unmodified data for SQI
        # Apply DWT
        _, cD = pywt.dwt(row, wavelet)
        transformed_row = cD.tolist()  # Concatenate coefficients
        row_min = min(transformed_row)
        row_max = max(transformed_row)
        if row_max != row_min:
            row = [(v - row_min) / (row_max - row_min) for v in transformed_row]
        else:
            row = [0 for _ in transformed_row]
        orig_len = len(row)  # save original length before padding
        if len(row) < max_columns:
            row.extend([0] * (max_columns - len(row)))  # Pad with zeros
        elif len(row) > max_columns:
            row = row[:max_columns]  # Truncate to max_columns
            orig_len = max_columns
        data.append(row)
        orig_lengths.append(orig_len)

# Convert the list of lists to a NumPy array
test_x = np.array(data)
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
test_x = torch.FloatTensor(test_x)

model.load_state_dict(torch.load('Save_Model/model_parameter-2023-5-31-1.pkl', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('Save_Model/model_parameter-2023-5-31-1.pkl'))
model.cpu()

val_outputs = model(test_x)['seg']
val_outputs = np.array(val_outputs.detach().numpy()).astype(np.int8)
val_outputs[val_outputs <= 0.5] = 0
val_outputs[val_outputs > 0.5] = 1

# Compute invalid rate (fraction of 0's) for each record
# Shape of val_outputs is [N, 1, max_columns].
# We'll sum across the channel (1) and samples (max_columns) dimensions, then divide by the total.
invalid_rates = []
for i in range(val_outputs.shape[0]):
    valid_length = orig_lengths[i]
    record_vals = val_outputs[i, 0, :valid_length]
    invalid_rate = (record_vals == 0).sum() / valid_length
    invalid_rates.append(invalid_rate)

# Compute SQI for each record using raw_signals
# Adjust "sampling_rate" and "method" as desired
sqi_values = []
for raw_signal in raw_signals:
    # Often nk.ppg_quality returns a dictionary or a DataFrame; take the mean or last value
    # Adjust indexing to fit your return structure, e.g. sqi_out['PPG_Quality'].mean().
    ppg_cleaned = nk.ppg_clean(raw_signal, sampling_rate=25)
    peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=25)
    quality_values = nk.ppg_quality(ppg_cleaned,
                                    ppg_pw_peaks=peaks_info["PPG_Peaks"],
                                    sampling_rate=25,
                                    method="templatematch")
    sqi_values.append(quality_values.mean())  # Example usage

# Count and print the number of records with invalid_rates > 0.5 and <= 0.5
num_greater_0_5 = sum(rate > 0.5 for rate in invalid_rates)
num_less_equal_0_5 = sum(rate <= 0.5 for rate in invalid_rates)
SQI_G065 = sum(rate > 0.65 for rate in sqi_values)
SQI_L065 = sum(rate <= 0.65 for rate in sqi_values)
print(f'Number of records with invalid_rates > 0.5: {num_greater_0_5}')
print(f'Number of records with invalid_rates <= 0.5: {num_less_equal_0_5}')
print(f'Number of records with SQI > 0.65: {SQI_G065}')
print(f'Number of records with SQI <= 0.65: {SQI_L065}')

# # # Sort records by their invalid_rates
sorted_indices = np.argsort(invalid_rates)

# # Plot the last three records with invalid_rates <= 0.5 (closest to 0.5)
print("Plotting records with invalid_rates <= 0.5:")
for i in sorted_indices:
    if sqi_values[i] > 0.65:
        ppg = test_x[i, 0, :orig_lengths[i]].numpy()
        preds = val_outputs[i, 0, :orig_lengths[i]]
        plt.figure()
        plt.plot(ppg, color='black')
        plt.fill_between(range(orig_lengths[i]), ppg, where=(preds == 0), color='red', alpha=0.3)
        plt.title(f'Record {i+1} - Invalid Rate: {invalid_rates[i]:.2f}, SQI: {sqi_values[i]:.2f}')
        plt.show()

# # Plot the last three records with invalid_rates <= 0.5 (closest to 0.5)
# print("Plotting records with invalid_rates <= 0.5:")
# for i in sorted_indices:
#     if invalid_rates[i] <= 0.5:
#         ppg = test_x[i, 0, :orig_lengths[i]].numpy()
#         preds = val_outputs[i, 0, :orig_lengths[i]]
#         plt.figure()
#         plt.plot(ppg, color='black')
#         plt.fill_between(range(orig_lengths[i]), ppg, where=(preds == 0), color='red', alpha=0.3)
#         plt.title(f'Record {i+1} - Invalid Rate: {invalid_rates[i]:.2f}, SQI: {sqi_values[i]:.2f}')
#         plt.show()

# # Plot the first three records with invalid_rates > 0.5 (closest to 0.5)
# print("Plotting records with invalid_rates > 0.5:")
# for i in sorted_indices:
#     if invalid_rates[i] > 0.5:
#         ppg = test_x[i, 0, :orig_lengths[i]].numpy()
#         preds = val_outputs[i, 0, :orig_lengths[i]]
#         plt.figure()
#         plt.plot(ppg, color='black')
#         plt.fill_between(range(orig_lengths[i]), ppg, where=(preds == 0), color='red', alpha=0.3)
#         plt.title(f'Record {i+1} - Invalid Rate: {invalid_rates[i]:.2f}')
#         plt.show()
