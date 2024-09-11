import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt

# Define a function to extract data from the zip files
def extract_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

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
    'Z014': 'Z/Z014.txt',
    'O014': 'O/O014.txt',
    'N014': 'N/N014.txt',
    'F014': 'F/F014.txt',
    'S014': 'S/S014.txt',
}

# Read the specified EEG files
sets = {}
for key, path in file_paths.items():
    try:
        sets[key] = read_eeg_file(path)
    except FileNotFoundError as e:
        print(e)
        sets[key] = None

# Create output directory if it doesn't exist
output_dir = "FFT_txt_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to save FFT results to a txt file
def save_fft_to_file(fft_freq, fft_data, output_path):
    # Combine the frequency and amplitude data
    fft_results = np.column_stack((fft_freq, np.abs(fft_data)))

    # Save to a txt file
    np.savetxt(output_path, fft_results, header="Frequency(Hz) Amplitude", fmt="%f %f")

# Define a function to compute FFT and plot the results with descriptions
def plot_and_save_fft(data, title, description, ax, output_path):
    n = len(data)
    fft_data = np.fft.fft(data)
    fft_freq = np.fft.fftfreq(n)

    # Only keep the positive part of the spectrum
    pos_mask = fft_freq > 0
    fft_data = fft_data[pos_mask]
    fft_freq = fft_freq[pos_mask]

    # Plot the FFT
    ax.plot(fft_freq, np.abs(fft_data), linewidth=0.5)
    ax.set_title(f'FFT of {title}\n{description}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    # Set x-axis limit to only show until 0.3
    ax.set_xlim(0, 0.3)

    # Set y-axis limit to the highest peak
    max_amplitude = np.max(np.abs(fft_data))
    ax.set_ylim(0, max_amplitude)

    # Save the FFT results to a text file
    save_fft_to_file(fft_freq, fft_data, output_path)

# Descriptions for each EEG set
descriptions = {
    'Z014': 'Set Z: Healthy volunteers with their eyes open.',
    'O014': 'Set O: Healthy volunteers with their eyes closed.',
    'N014': 'Set N: Patients during seizure-free intervals, comparing the hippocampal formation to the epileptogenic zone.',
    'F014': 'Set F: Patients within an epileptogenic zone during seizure-free intervals.',
    'S014': 'Set S: Patients during seizure activity.',
}

# Output file paths for FFT results
output_file_paths = {
    'Z014': os.path.join(output_dir, 'FFT_output_Z014.txt'),
    'O014': os.path.join(output_dir, 'FFT_output_O014.txt'),
    'N014': os.path.join(output_dir, 'FFT_output_N014.txt'),
    'F014': os.path.join(output_dir, 'FFT_output_F014.txt'),
    'S014': os.path.join(output_dir, 'FFT_output_S014.txt'),
}

# Plot FFT graphs for all the samples and save results to txt files
fig, axs = plt.subplots(len(sets), 1, figsize=(15, 10))

# Plot and save each FFT graph
for ax, (key, data) in zip(axs, sets.items()):
    if data is not None:
        output_path = output_file_paths[key]
        plot_and_save_fft(data, key, descriptions[key], ax, output_path)

plt.tight_layout()
plt.show()
