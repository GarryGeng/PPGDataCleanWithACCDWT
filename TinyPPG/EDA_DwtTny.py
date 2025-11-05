import numpy as np
# from utils import *
import matplotlib.pyplot as plt
import torch
from scipy.signal import find_peaks
from model import Model
import pywt  # Added import for pywt
import neurokit2 as nk
import pandas as pd

model = Model()


def plot_all_categories(df, test_x, val_outputs, orig_lengths):
    columns = df.columns
    threshold = 15
    range_idx = 4
    merged_len = 65

    for i in range(min(5, len(df))):
        if all(x in columns for x in ['acc_x','acc_y','acc_z']):
            data_x = [float(x) for x in df['acc_x'].iloc[i].split(',')]
            data_y = [float(y) for y in df['acc_y'].iloc[i].split(',')]
            data_z = [float(z) for z in df['acc_z'].iloc[i].split(',')]
            combined_acc = [
                (data_x[idx]**2 + data_y[idx]**2 + data_z[idx]**2)**0.5
                for idx in range(len(data_x))
            ]
            changes = [False]*len(combined_acc)
            for idx in range(range_idx, len(combined_acc)):
                prev_avg = sum(combined_acc[idx-range_idx:idx]) / range_idx
                if abs(combined_acc[idx] - prev_avg) > threshold:
                    changes[idx] = True

            intervals = []
            start = None
            for idx in range(len(changes)):
                if changes[idx] and start is None:
                    start = idx
                elif not changes[idx] and start is not None:
                    intervals.append((start, idx - 1))
                    start = None
            if start is not None:
                intervals.append((start, len(changes) - 1))

            acc_intervals = []
            for (s, e) in intervals:
                if not acc_intervals:
                    acc_intervals.append((s, e))
                else:
                    last_s, last_e = acc_intervals[-1]
                    if s - last_e <= merged_len:
                        acc_intervals[-1] = (last_s, max(e, last_e))
                    else:
                        acc_intervals.append((s, e))

        plt.figure()
        if 'ppg' in columns:
            ppg_data = [float(x) for x in df['ppg'].iloc[i].split(',')]
            plt.plot(ppg_data, label=f'PPG (Record {i+1})')
            for (start_idx, end_idx) in acc_intervals:
                plt.axvspan(start_idx, end_idx, color='red', alpha=0.2)

            # Overlay model prediction
            preds = val_outputs[i, 0, :orig_lengths[i]]
            min_len = min(len(ppg_data), orig_lengths[i])
            plt.fill_between(
                range(min_len),
                ppg_data[:min_len],
                where=(preds[:min_len] == 0),
                color='blue',
                alpha=0.3,
                label='Model Pred'
            )
        plt.title(f"All Categories + Model Predictions - Record {i+1}")
        plt.legend()
        plt.show()

# Commented out to avoid parsing error
df = pd.read_csv('PPG_ACC.csv')  # Contains columns: ppg,acc_x,acc_y,acc_z
all_ppg = []
for row in df['ppg']:
    split_vals = [int(x) for x in row.split(',') if x.strip()]
    # Change extend to append to store each split_vals as a list
    all_ppg.append(split_vals)

# Update processing to handle all_ppg as a list of lists
data = []
raw_signals = []
orig_lengths = []
max_columns = 1920

for split_ppg in all_ppg:
    raw_signals.append(split_ppg)  # Keep original data for SQI
    wavelet = pywt.Wavelet('coif1')
    _, cD = pywt.dwt(split_ppg, wavelet)
    transformed_row = cD.tolist()
    row_min = min(transformed_row)
    row_max = max(transformed_row)
    if row_max != row_min:
        norm_row = [(v - row_min) / (row_max - row_min) for v in transformed_row]
    else:
        norm_row = [0 for _ in transformed_row]
    
    orig_len = len(norm_row)
    if orig_len < max_columns:
        norm_row.extend([0] * (max_columns - orig_len))
    else:
        norm_row = norm_row[:max_columns]
        orig_len = max_columns
    
    data.append(norm_row)
    orig_lengths.append(orig_len)

device = torch.device("mps")
test_x = np.array(data).reshape((len(data), 1, max_columns))
test_x = torch.FloatTensor(test_x)
test_x = test_x.to(device)
model.load_state_dict(torch.load('Save_Model/model_parameter-2023-5-31-1.pkl', map_location=torch.device('mps')))
model.to(device)

val_outputs = model(test_x)['seg']
# val_outputs = np.array(val_outputs.detach().numpy()).astype(np.int8)
val_outputs = np.array(val_outputs.detach().cpu().numpy()).astype(np.int8)
val_outputs[val_outputs <= 0.5] = 0
val_outputs[val_outputs > 0.5] = 1

invalid_rates = []
for i in range(val_outputs.shape[0]):
    valid_length = orig_lengths[i]
    record_vals = val_outputs[i, 0, :valid_length]
    invalid_rate = (record_vals == 0).sum() / valid_length
    invalid_rates.append(invalid_rate)

sqi_values = []
for signal in raw_signals:
    ppg_cleaned = nk.ppg_clean(signal, sampling_rate=25)
    peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=25)
    quality_values = nk.ppg_quality(ppg_cleaned,
                                    ppg_pw_peaks=peaks_info["PPG_Peaks"],
                                    sampling_rate=25,
                                    method="templatematch")
    sqi_values.append(quality_values.mean())

num_greater_0_5 = sum(rate > 0.5 for rate in invalid_rates)
num_less_equal_0_5 = sum(rate <= 0.5 for rate in invalid_rates)
SQI_G065 = sum(rate > 0.65 for rate in sqi_values)
SQI_L065 = sum(rate <= 0.65 for rate in sqi_values)
print(f'Number of records with invalid_rates > 0.5: {num_greater_0_5}')
print(f'Number of records with invalid_rates <= 0.5: {num_less_equal_0_5}')
print(f'Number of records with SQI > 0.65: {SQI_G065}')
print(f'Number of records with SQI <= 0.65: {SQI_L065}')

sorted_indices = np.argsort(invalid_rates)

print("Showing EDA_all_TinyPPG Categories:")
df_all = pd.read_csv('PPG_ACC.csv')
plot_all_categories(df_all, test_x, val_outputs, orig_lengths)