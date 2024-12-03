import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# region Formula 1- Average deviation of amplitudes

def calculate_mx_j(segment):
    N_star = len(segment)
    x_mean = np.mean(segment)
    mx_j = np.sum(np.abs(segment - x_mean)) / N_star
    return mx_j

def get_all_mxj(data):
    num_segments = 16
    segment_length = len(data) // num_segments
    mx_values = []
    for j in range(num_segments):
        segment = data[j * segment_length:(j + 1) * segment_length]
        mx_j = calculate_mx_j(segment)
        mx_values.append(mx_j)

    return mx_values

data = np.loadtxt('C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_Z014.txt')
print("Formula 1: All mx_j = ");
print(get_all_mxj(data));

# endregion

# region Formula 2- Center Frequency
# First we need to convert the data to FFT:

# next we will calculate the  Center Frequency
# N∗ is the length of the subsegment.
# 𝜔𝑖 represents the frequency associated with the i-th FFT component.
# S(ωi) is the amplitude of the Fourier transform at the i-th frequency.

def calculate_mj_omega(segment_fft):
    amplitudes = segment_fft['Amplitude']
    N_star = len(amplitudes)
    S_w = np.abs(amplitudes)
    frequencies = np.fft.fftfreq(N_star)
    half_N_star = N_star // 2
    S_w_half = S_w[:half_N_star]
    frequencies_half = frequencies[:half_N_star]
    m_j_omega = (2 / N_star) * np.sum(frequencies_half * S_w_half / np.mean(S_w_half))

    return m_j_omega

def get_all_mwj(fft_data):
    num_segments = 16
    segment_length = len(fft_data) // num_segments
    center_frequencies = []

    for j in range(num_segments):
        segment_fft = fft_data[j * segment_length:(j + 1) * segment_length]
        m_j_omega = calculate_mj_omega(segment_fft)
        center_frequencies.append(m_j_omega)

    return center_frequencies

fft_data = np.loadtxt('FFT_output_Z014.txt', skiprows=1, dtype=[('Frequency', 'float64'), ('Amplitude', 'float64')])
center_frequencies = get_all_mwj(fft_data)

np.savetxt('Z014_center_frequencies_fft.txt', center_frequencies)

print("Center frequencies for all segments saved to Z014_center_frequencies_fft.txt")

# endregion

# region Formula 3- Fx, Variance in the time space
m_x_values = get_all_mxj(data)
mean_m_x = np.mean(m_x_values)
F_x = np.mean(np.abs(m_x_values - mean_m_x))
print("F_x", F_x)
# endregion

# region Formula 4- Fw, Variance in the frequency space
m_w_values = get_all_mwj(fft_data)
mean_m_w = np.mean(m_w_values)
F_w = np.mean(np.abs(m_w_values - mean_m_w))
print("F_w", F_w)

# endregion

 # region Formula 5-Correlation

# #C(E,N)
# def correlation_sum(X, epsilon):
#     N = len(X)
#     T = 5
#     sum_total = 0
#     for i in range(N - T):
#         for j in range(i + T, N):
#             distance = np.linalg.norm(X[i] - X[j])
#             # print("distance:", distance)
#             if epsilon > distance:
#                 sum_total += 1
#     normalization = 2 / ((N - T) * (N - T - 1))
#     return normalization * sum_total

#נגזרת

# def correlation_sum(X, epsilon):
#     N = len(X)
#     T = 5
#     # Compute pairwise distances
#     distances = cdist(X[:-T].reshape(-1, 1), X[T:].reshape(-1, 1), 'euclidean')
#     count = np.sum(distances < epsilon)
#     normalization = 2 / ((N - T) * (N - T - 1))
#     return normalization * count
def correlation_sum(X, epsilon):
    N = len(X)
    T = 5
    sum_total = 0
    for i in range(N - T):
        for j in range(i + T, N):
            distance = round(np.linalg.norm(X[i] - X[j]), 2)  # Round distance to 2 decimal places
            if epsilon > distance:
                sum_total += 1
    normalization = 2 / ((N - T) * (N - T - 1))
    return normalization * sum_total

def smooth_heaviside(x, epsilon, steepness=1):
    return 1 / (1 + np.exp(-steepness * x / epsilon))  # Sigmoid approximation

def partial_correlation_sum(X, epsilon):
    N = len(X)
    T = 5
    sum_total = 0
    epsilon=round(epsilon,3)
    for i in range(N - T):
        for j in range(i + T, N):
            distance = round(np.linalg.norm(X[i] - X[j]), 3)  # Round distance to 2 decimal places
            # print("distance:", distance)
            if abs(distance - epsilon) == 0:
                sum_total += 1
    normalization = 2 / ((N - T) * (N - T - 1))
    return normalization * sum_total

# def partial_correlation_sum(X, epsilon):
#     N = len(X)
#     T = 5
#     correlation_sum = 0
#
#     for i in range(N - 1):
#         for j in range(i + T, N - 1):
#             distance = np.linalg.norm(data[i] - data[j])
#             correlation_sum += smooth_heaviside(epsilon - distance, epsilon)
#
#     # Normalizing
#     correlation_sum *= (2 / ((N - T) * (N - T - 1)))
#     return correlation_sum

    # distances = cdist(X[:-T].reshape(-1, 1), X[T:].reshape(-1, 1), 'euclidean')
    # count = np.sum(np.abs(distances - epsilon) == 0)
    # normalization = 2 / ((N - T) * (N - T - 1))
    # return normalization * count

# def partial_correlation_sum(X, epsilon):
#     N = len(X)
#     T = 5
#     sum_total = 0
#     for i in range(N - T):
#         for j in range(i + T, N):
#             distance = np.linalg.norm(X[i] - X[j])
#             # זה השורה שצריך לוודא עם גיא
#             if abs(distance - epsilon) == 0:
#                 sum_total += 1
#     normalization = 2 / ((N - T) * (N - T - 1))
#     return normalization * sum_total

#d(e,n)
def calculate_d(X, epsilon):
   c_sum=correlation_sum(X,epsilon)
   # Calculate the partial correlation sum
   c_derivative=partial_correlation_sum(X,epsilon)
   if c_sum == 0:
       return np.nan
   derivative = c_derivative / c_sum
   print(f"Epsilon: {epsilon}, Correlation Sum: {c_sum}, Partial Correlation Sum: {c_derivative}")
   return derivative*epsilon

def run_calculate_d_on_epsilon_range(data):
    # Generate epsilon values: 2^-8, 2^-7, ..., 2^-1, 0
    #epsilon_values = np.array([2**-i for i in range(8, 0, -1)] + [0])
    epsilon_values = np.logspace(-2, 0, num = 50)  # Larger range for epsilon
    #epsilon_values = np.linspace(0.1, 2, 100)  # Adjust these values
    results = []

    for epsilon in epsilon_values:
        if epsilon > 0:  # Ensure we don't run into division by zero
            print(f"Calculating for epsilon = {epsilon}")
            result = calculate_d(data, epsilon)
            results.append(result)
        else:
            results.append(np.nan)  # Handle zero epsilon separately

    return epsilon_values, results

# List of file paths
file_paths = [
    'C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_Z014.txt',
    'C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_F014.txt',
    'C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_O014.txt',
    'C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_S014.txt',
    'C:/Users/shire/PycharmProjects/pythonProject/EEG-Time-Series/FFT_txt_output/FFT_output_N014.txt'
]

# Loop over all datasets and generate plots
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    data = np.loadtxt(file_path, skiprows=1, usecols=1)  # Only load the second column (Amplitude)

    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Run the calculation
    epsilon_values, results = run_calculate_d_on_epsilon_range(data)

    # Plot results for each dataset individually
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, results, marker='o', label=f'{file_path.split("/")[-1]}')
    plt.xlabel('Epsilon')
    plt.ylabel('d(E, N)')
    plt.title(f'd(E, N) vs. Epsilon for {file_path.split("/")[-1]}')
    plt.grid(True)
    plt.legend(loc='best')
    # Save individual plot
    individual_plot_filename = f'{file_path.split("/")[-1].split(".")[0]}_plot.png'
    plt.savefig(individual_plot_filename)
    print(f"Saved plot for {file_path} as {individual_plot_filename}")
    plt.close()

# Final plot with all datasets together
plt.figure(figsize=(10, 6))
for file_path in file_paths:
    data = np.loadtxt(file_path, skiprows=1, usecols=1)  # Only load the second column (Amplitude)

    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Run the calculation
    epsilon_values, results = run_calculate_d_on_epsilon_range(data)

    # Plot results for each dataset
    plt.plot(epsilon_values, results, marker='o', label=f'{file_path.split("/")[-1]}')

plt.xlabel('Epsilon')
plt.ylabel('d(E, N)')
plt.title('Plot of d(E, N) vs. Epsilon for Different Datasets')
plt.grid(True)
plt.legend(loc='best')

# Save the final combined plot
final_plot_filename = 'combined_plot.png'
plt.savefig(final_plot_filename)
print(f"Saved final combined plot as {final_plot_filename}")
plt.show()
#endregion