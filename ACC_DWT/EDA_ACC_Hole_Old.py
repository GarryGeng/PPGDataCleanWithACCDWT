import matplotlib.pyplot as plt
import pandas as pd
import pywt  # Added import for pywt
import numpy as np
import neurokit2 as nk

def plot_all_categories():
    df = pd.read_csv('PPG_ACC.csv')
    columns = df.columns  # Include all columns including PPG
    
    # threshold = 15
    threshold = 40
    range_idx = 2
    merged_len = 65
    removed_len = 3

    # Initialize lists for raw signals and transformed data
    raw_signals = []
    transformed_signals = []
    orig_lengths = []

    wavelet = pywt.Wavelet('coif1')  # Define wavelet
    # wavelet = pywt.Wavelet('db4')

    for i in range(min(5, len(df))):
        acc_intervals = []
        plt.figure(figsize=(12, 10))
        
        # Combine 'acc_x','acc_y','acc_z' for change detection
        if all(x in columns for x in ['acc_x','acc_y','acc_z']):
            data_x = df['acc_x'].iloc[i].split(',')
            data_y = df['acc_y'].iloc[i].split(',')
            data_z = df['acc_z'].iloc[i].split(',')

            data_x = [float(x) for x in data_x]
            data_y = [float(y) for y in data_y]
            data_z = [float(z) for z in data_z]

            # Downsample accelerometer data by taking every second sample
            data_x = data_x[::2]
            data_y = data_y[::2]
            data_z = data_z[::2]

            # combined_acc = [
            #     (data_x[idx]**2 + data_y[idx]**2 + data_z[idx]**2)**0.5
            #     for idx in range(len(data_x))
            # ]
            # # Perform change detection on combined_acc
            # changes = [False] * len(combined_acc)
            # for idx in range(range_idx, len(combined_acc)):
            #     prev_avg = sum(combined_acc[idx-range_idx:idx]) / range_idx
            #     if abs(combined_acc[idx] - prev_avg) > threshold:
            #         changes[idx] = True

            n = len(data_x)
            changes = [False] * n
            for idx in range(range_idx, n):
                # 计算各轴变化幅度
                delta_x = abs(data_x[idx] - np.mean(data_x[idx - range_idx:idx]))
                delta_y = abs(data_y[idx] - np.mean(data_y[idx - range_idx:idx]))
                delta_z = abs(data_z[idx] - np.mean(data_z[idx - range_idx:idx]))

                # 组合变化幅度（可以根据需要调整权重）
                combined_delta = delta_x + delta_y + delta_z
                # 也可以用其他组合方式，例如： combined_delta = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
        
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
        
        for j, col in enumerate(columns):
            data_values = df[col].iloc[i].split(',')
            data_values = [int(x) for x in data_values]
            
            if col == 'ppg':
                # Convert PPG signal to list of integers
                row = [int(x) for x in data_values]
                raw_signals.append(row)  # Keep a copy of the unmodified data for SQI

                # Apply DWT
                _, cD = pywt.dwt(row, wavelet)
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

                # Cal SQI
                normalized_row_np = np.array(normalized_row)
                sqi_val = nk.ppg_quality(
                    normalized_row_np,
                    ppg_pw_peaks=nk.ppg_findpeaks(normalized_row_np, sampling_rate=12.5)["PPG_Peaks"],
                    sampling_rate=12.5,
                    method="templatematch"
                ).mean()
                print(f"SQI for Record {i+1}: {sqi_val}")
            else:
                # Downsample acc data for plotting
                data_values = data_values[::2]  # Take every second sample
                plot_data = data_values
                label = f'Downsampled {col} (Record {i+1})'
            
            plt.subplot(5, 1, j + 1)
            plt.plot(plot_data, label=label)
            plt.legend()

        # Highlight merged acc_intervals in PPG (if 'ppg' is present)
        if 'ppg' in columns:
            ppg_index = list(columns).index('ppg')
            plt.subplot(5, 1, ppg_index + 1)
            for (start_idx, end_idx) in acc_intervals:
                plt.axvspan(start_idx, end_idx, color='red', alpha=0.2)

        plt.tight_layout()
        plt.show()

plot_all_categories()
