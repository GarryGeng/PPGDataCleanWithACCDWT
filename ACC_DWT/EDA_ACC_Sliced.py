import matplotlib.pyplot as plt
import pandas as pd
import pywt  # Added import for pywt
import numpy as np
import neurokit2 as nk

def plot_all_categories():
    df = pd.read_csv('PPG_ACC_New.csv')
    columns = df.columns  # Include all columns including PPG
    
    # threshold = 15
    threshold = 40
    range_idx = 2
    merged_len = 65
    removed_len = 1

    # Initialize lists for raw signals and transformed data
    raw_signals = []
    transformed_signals = []
    orig_lengths = []

    wavelet = pywt.Wavelet('coif1')  # Define wavelet
    # wavelet = pywt.Wavelet('db4')

    for i in range(min(5, len(df))):
        acc_intervals = []
        plt.figure(figsize=(12, 5))
        
        # Combine 'acc_x','acc_y','acc_z' for change detection
        if all(x in columns for x in ['acc_x','acc_y','acc_z']):
            data_x = df['acc_x'].iloc[i].split(',')
            data_y = df['acc_y'].iloc[i].split(',')
            data_z = df['acc_z'].iloc[i].split(',')

            data_x = [float(x) for x in data_x]
            data_y = [float(y) for y in data_y]
            data_z = [float(z) for z in data_z]

            data_x = [val for x in data_x for val in (x, x, x, x)]
            data_y = [val for y in data_y for val in (y, y, y, y)]
            data_z = [val for z in data_z for val in (z, z, z, z)]

            n = len(data_x)
            changes = [False] * n
            for idx in range(range_idx, n):
                delta_x = abs(data_x[idx] - np.mean(data_x[idx - range_idx:idx]))
                delta_y = abs(data_y[idx] - np.mean(data_y[idx - range_idx:idx]))
                delta_z = abs(data_z[idx] - np.mean(data_z[idx - range_idx:idx]))

                combined_delta = delta_x + delta_y + delta_z
        
                if combined_delta > threshold:
                    changes[idx] = True

            intervals = []
            start = None
            # remove the first range_idx samples
            intervals.append((0, range_idx))
            for idx in range(len(changes)):
                if changes[idx] and start is None:
                    start = idx
                elif not changes[idx] and start is not None:
                    intervals.append((start, idx - 1))
                    start = None
            if start is not None:
                intervals.append((start, len(changes) - 1))

            # Use merged intervals to update acc_intervals
            for (s, e) in intervals:
                if not acc_intervals:
                    acc_intervals.append((s, e))
                else:
                    last_s, last_e = acc_intervals[-1]
                    if s - last_e <= merged_len:
                        acc_intervals[-1] = (last_s, max(e, last_e))
                    else:
                        acc_intervals.append((s, e))

            # Remove short intervals
            acc_intervals = [(s, e) for (s, e) in acc_intervals if (e - s) >= removed_len]
        
        # Filter out accelerometer columns for plotting
        plot_columns = [col for col in columns if col not in ['acc_x','acc_y','acc_z']]
        for j, col in enumerate(plot_columns):

            data_values = df[col].iloc[i].split(',')
            data_values = [int(x) for x in data_values]
            
            if col == 'ppg':
                # Convert PPG signal to list of integers
                row = [int(x) for x in data_values]
                raw_signals.append(row)  # Keep a copy of the unmodified data for SQI

                # Remove intervals from PPG row
                removal_set = set()
                for (s, e) in acc_intervals:
                    for idx2 in range(s, e + 1):
                        removal_set.add(idx2)
                cleaned_row = [val for idx2, val in enumerate(row) if idx2 not in removal_set]

                # Apply DWT on cleaned_row
                _, cD = pywt.dwt(cleaned_row, wavelet)
                transformed_row = cD.tolist()

                row_min = min(transformed_row)
                row_max = max(transformed_row)
                if row_max != row_min:
                    normalized_row = [(v - row_min) / (row_max - row_min) for v in transformed_row]
                else:
                    normalized_row = [0 for _ in transformed_row]
                
                orig_len = len(normalized_row)  # Save original length

                # Append transformed data without padding
                transformed_signals.append(normalized_row)
                orig_lengths.append(orig_len)

                # Use transformed data for plotting
                plot_data = normalized_row
                label = f'DWT PPG (Record {i+1})'
            else:
                # Downsample acc data for plotting
                data_values = data_values[::2]  # Take every second sample
                plot_data = data_values
                label = f'Downsampled {col} (Record {i+1})'
            
            plt.subplot(len(plot_columns), 1, j + 1)
            plt.plot(plot_data, label=label)
            plt.legend()

        plt.tight_layout()
        plt.show()

plot_all_categories()
