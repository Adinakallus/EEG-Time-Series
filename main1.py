import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt

# Define a function to extract data from the zip files
def extract_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Extract data from each set
extract_data('Z.zip', 'Z')
extract_data('O.zip', 'O')

# Define a function to read EEG files
def read_eeg_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Convert to a numpy array of floats
    data = np.array([float(sample.strip()) for sample in data])
    return data

# Define file paths for the specified EEG files
file_paths = {
    'Z061': 'Z/Z061.txt',
    'Z063': 'Z/Z063.txt',
    'O061': 'O/O061.txt',
    'O063': 'O/O063.txt'
}

# Read the specified EEG files
sets = {}
for key, path in file_paths.items():
    try:
        sets[key] = read_eeg_file(path)
    except FileNotFoundError as e:
        print(e)
        sets[key] = None

# Define the number of samples to plot (1 second worth of data)
num_samples = int(173.61)

# Plot the first 1 second of samples from the specified EEG segments
plt.figure(figsize=(15, 10))

for i, (key, data) in enumerate(sets.items(), start=1):
    if data is not None:
        plt.subplot(4, 1, i)
        plt.plot(data[:num_samples])  # Plot the first 1 second of samples
        plt.title(f'EEG Segment - {key}')
        plt.ylabel('Amplitude')
        if i == 4:
            plt.xlabel('Time (samples)')

plt.tight_layout()
plt.show()
