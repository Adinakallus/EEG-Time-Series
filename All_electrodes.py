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

# Define a function to plot and save EEG data
def plot_and_save_eeg(sets, output_folder, electrode_number):
    plt.figure(figsize=(15, 10))

    for i, (key, data) in enumerate(sets.items(), start=1):
        if data is not None:
            plt.subplot(5, 1, i)  # Changed to 5 subplots to match the number of datasets
            plt.plot(data, linewidth=0.5)  # Set the line width to 0.5
            plt.title(f'EEG Segment - {key}')
            plt.ylabel('Amplitude')
            if i == 5:
                plt.xlabel('Time (samples)')

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'Electrode_{electrode_number:03d}.png'))
    plt.close()

# Loop through electrode numbers from 1 to 100, read the data, and save the plots
for electrode_number in range(1, 101):
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

    # Plot and save the EEG data for this electrode
    output_folder = 'EEG_Plots'
    plot_and_save_eeg(sets, output_folder, electrode_number)
