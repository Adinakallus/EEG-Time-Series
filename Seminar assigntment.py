import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

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


data = np.loadtxt('C:/Users/adina/Downloads/Seminar/Seminar/Z/Z001.txt')
print(get_all_mxj(data))

# endregion

# region Formula 2- Center Frequency
# First we need to convert the data to FFT:


# next we will calculate the  Center Frequency
# N∗ is the length of the subsegment.
# 𝜔𝑖 represents the frequency associated with the i-th FFT component.
# S(ωi) is the amplitude of the Fourier transform at the i-th frequency.


def calculate_mj_omega(segment_fft):
    N_star = len(segment_fft)
    S_w = np.abs(segment_fft)

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


fft_data = np.loadtxt('Z001_fft.txt', dtype=complex)

center_frequencies = get_all_mwj(fft_data)

np.savetxt('Z001_center_frequencies_fft.txt', center_frequencies)

print("Center frequencies for all segments saved to Z001_center_frequencies_fft.txt")

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


#C(E,N)
def correlation_sum(X, epsilon):
    N = len(X)
    T = 5
    sum_total = 0
    for i in range(N - T):
        for j in range(i + T, N):
            distance = np.linalg.norm(X[i] - X[j])
            # print("distance:", distance)
            if epsilon > distance:
                sum_total += 1
    normalization = 2 / ((N - T) * (N - T - 1))
    return normalization * sum_total

#נגזרת
def partial_correlation_sum(X, epsilon):
    N = len(X)
    T = 5
    sum_total = 0
    for i in range(N - T):
        for j in range(i + T, N):
            distance = np.linalg.norm(X[i] - X[j])
            # זה השורה שצריך לוודא עם גיא
            if abs(distance - epsilon) == 0:
                sum_total += 1
    normalization = 2 / ((N - T) * (N - T - 1))
    return normalization * sum_total

#d(e,n)
def calculate_d(X, epsilon):
   c_sum=correlation_sum(X,epsilon)
   # Calculate the partial correlation sum
   c_derivative=partial_correlation_sum(X,epsilon)
   if c_sum == 0:
       return np.nan

   derivative = c_derivative / c_sum
   return derivative*epsilon

def run_calculate_d_on_epsilon_range(data):
    # Generate epsilon values: 2^-8, 2^-7, ..., 2^-1, 0
    epsilon_values = np.array([2**-i for i in range(8, 0, -1)] + [0])
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
    'C:/Users/adina/Downloads/Seminar/Seminar/Z/Z001.txt',
    'C:/Users/adina/Downloads/Seminar/Seminar/F/F001.txt',
    'C:/Users/adina/Downloads/Seminar/Seminar/O/O001.txt',
    'C:/Users/adina/Downloads/Seminar/Seminar/S/S001.txt',
    'C:/Users/adina/Downloads/Seminar/Seminar/N/N001.txt'
]

# Loop over all datasets and generate plots
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    data = np.loadtxt(file_path)

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
    data = np.loadtxt(file_path)

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

# def calculate_d(X, epsilon):
#     # Calculate C(E, N)
#     C_value = correlation_sum(X, epsilon)
#
#     # Calculate C(E, N)'/E' using the approximation
#     partial_C_value = partial_correlation_sum(X, epsilon)
#
#     if C_value == 0:
#         print("Warning: C(E, N) is zero.")
#         return np.nan
#
#     # Calculate d(E, N) = (C(ε, N)'/E') / C(E, N)
#     derivative = partial_C_value / C_value
#     return derivative
#
#
# x = data.reshape(1, -1)
# print("reshaped data:", data)
#
#
# # endregion
# def run_calculate_d_on_epsilon_range(data):
#     epsilon_values = np.linspace(1/256, 0, num=100)  # 100 points from 1/256 to 0
#     results = []
#
#     for epsilon in epsilon_values:
#         if epsilon > 0:  # Ensure we don't run into division by zero
#             result = calculate_d(data, epsilon)
#             results.append(result)
#         else:
#             results.append(np.nan)  # Handle zero epsilon separately
#
#     return epsilon_values, results
#
# plt.figure(figsize=(10, 6))
#
# # Loop over all datasets
# file_paths = [
#     'C:/Users/adina/Downloads/Seminar/Seminar/Z/Z001.txt',
#     'C:/Users/adina/Downloads/Seminar/Seminar/F/F001.txt',
#     'C:/Users/adina/Downloads/Seminar/Seminar/O/O001.txt',
#     'C:/Users/adina/Downloads/Seminar/Seminar/S/S001.txt',
#     'C:/Users/adina/Downloads/Seminar/Seminar/N/N001.txt'
# ]
# for file_path in file_paths:
#     data = np.loadtxt(file_path)
#
#     # Normalize the data
#     data = (data - np.min(data)) / (np.max(data) - np.min(data))
#
#     # Run the calculation
#     epsilon_values, result = run_calculate_d_on_epsilon_range(data)
#
#     # Plot results for each dataset
#     plt.plot(epsilon_values, result, marker='o', label=f'{file_path.split("/")[-1]}')
#
# plt.xlabel('Epsilon')
# plt.ylabel('d(E, N)')
# plt.title('Plot of d(E, N) vs. Epsilon for Different Datasets')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()
#
