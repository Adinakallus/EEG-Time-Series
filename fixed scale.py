import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt

# Define a function to extract data from the zip files
def extract_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Extract data from each set
extract_data('N.zip', 'N')
extract_data('F.zip', 'F')
extract_data('O.zip', 'O')
extract_data('S.zip', 'S')
extract_data('Z.zip', 'Z')

# Define a function to read EEG files
def read_eeg_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Convert to a numpy array of floats
    data = np.array([float(sample.strip()) for sample in data])
    return data

# Define a function to compute the FFT of EEG data
def compute_fft(data, sampling_rate):
    N = len(data)
    T = 1.0 / sampling_rate
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, T)[:N//2]
    yf = 2.0/N * np.abs(yf[:N//2])
    return xf, yf

# Define a function to plot and save EEG data
def plot_eeg(sets, output_folder, electrode_number):
    plt.figure(figsize=(15, 20))

    for i, (key, data) in enumerate(sets.items(), start=1):
        if data is not None:
            plt.subplot(5, 1, i)
            plt.plot(data, linewidth=0.5)
            plt.title(f'EEG Segment - {key}')
            plt.ylabel('Amplitude')
            if i == 5:
                plt.xlabel('Time (samples)')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'Electrode_{electrode_number:03d}_EEG.png'))
    plt.close()

# Define a function to plot and save FFT data
def plot_fft(sets, output_folder, electrode_number, sampling_rate):
    plt.figure(figsize=(15, 20))

    for i, (key, data) in enumerate(sets.items(), start=1):
        if data is not None:
            xf, yf = compute_fft(data, sampling_rate)
            plt.subplot(5, 1, i)
            plt.plot(xf, yf, linewidth=0.5)
            plt.title(f'FFT of EEG Segment - {key}')
            plt.ylabel('Amplitude')
            if i == 5:
                plt.xlabel('Frequency (Hz)')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'Electrode_{electrode_number:03d}_FFT.png'))
    plt.close()

# Electrode number to be displayed
electrode_number = 37
sampling_rate = 100  # Assumed sampling rate

file_paths = {
    f'Z{electrode_number:03d}': f'Z/Z{electrode_number:03d}.txt',
    f'O{electrode_number:03d}': f'O/O{electrode_number:03d}.txt',
    f'N{electrode_number:03d}': f'N/N{electrode_number:03d}.txt',
    f'F{electrode_number:03d}': f'F/F{electrode_number:03d}.txt',
    f'S{electrode_number:03d}': f'S/S{electrode_number:03d}.txt'
}

# Read the specified EEG files
sets = {}
for key, path in file_paths.items():
    try:
        sets[key] = read_eeg_file(path)
    except FileNotFoundError as e:
        print(e)
        sets[key] = None

# Plot and save all EEG data in one file and all FFT data in another file for this electrode
output_folder = 'EEG_Plots'
plot_eeg(sets, output_folder, electrode_number)
plot_fft(sets, output_folder, electrode_number, sampling_rate)
