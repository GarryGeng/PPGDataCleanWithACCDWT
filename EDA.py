import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks

def plot_coif1_dwt():
    with open('PPGData.txt', 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    wavelet = pywt.Wavelet('coif1')
    for i in range(min(5, len(lines))):
        data_values = [float(x) for x in lines[i].replace('"','').split(',')]
        cA, cD = pywt.dwt(data_values, wavelet)
        plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(cA, label=f'Approx Coeffs (Line {i+1})')
        # peaks_cA, _ = find_peaks(cA, height=0.5)  # Adjust the height parameter as needed
        # plt.plot(peaks_cA, cA[peaks_cA], 'ro')
        # plt.legend()
        
        plt.subplot(2,1,2)
        plt.plot(cD, label=f'Detail Coeffs (Line {i+1})')
        peaks_cD, _ = find_peaks(cD, distance=7)  # Adjust the height parameter as needed
        plt.plot(peaks_cD, cD[peaks_cD], 'ro')
        plt.legend()
        
        plt.show()

plot_coif1_dwt()
