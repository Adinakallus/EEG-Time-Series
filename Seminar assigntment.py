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


def partial_correlation_sum(X, epsilon):
    N = len(X)
    T = 5
    sum_total = 0
    for i in range(N - T):
        for j in range(i + T, N):
            distance = np.linalg.norm(X[i] - X[j])
            # print("distance:", distance)
            if abs(distance - epsilon) == 0:
                sum_total += 1
    normalization = 2 / ((N - T) * (N - T - 1))
    return normalization * sum_total


def calculate_d(X, epsilon):
    # Calculate C(E, N)
    C_value = correlation_sum(X, epsilon)

    # Calculate C(E, N)'/E' using the approximation
    partial_C_value = partial_correlation_sum(X, epsilon)

    if C_value == 0:
        print("Warning: C(E, N) is zero.")
        return np.nan

    # Calculate d(E, N) = (C(ε, N)'/E') / C(E, N)
    derivative = partial_C_value / C_value
    return derivative


x = data.reshape(1, -1)
print("reshaped data:", data)


# endregion

def run_calculate_d_on_epsilon_range(data):
    results = []
    epsilon_values = np.arange(0, 1.1, 0.1)  # Epsilon runs from 0 to 1 with a step of 0.1

    for epsilon in epsilon_values:
        if epsilon > 0:  # Ensure we're not taking log2 of zero
            # log2_epsilon = np.log2(epsilon)
            # if np.isfinite(log2_epsilon):  # Check if log2 is a valid number
            result = calculate_d(data, epsilon)
            print("result for epsilon =", epsilon, "is:", result)
            results.append(result)

        else:
            results.append(np.nan)  # For epsilon = 0, result is undefined

    return epsilon_values, results


data = (data - np.min(data)) / (np.max(data) - np.min(data))
print("data:", data)
# Assuming `data` is your input dataset
epsilon_values, result = run_calculate_d_on_epsilon_range(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, result, marker='o', color='b', label='d(E, N)')
plt.xlabel('Epsilon')
plt.ylabel('d(E, N)')
plt.title('Plot of d(E, N) vs. Epsilon')
plt.grid(True)
plt.legend()
plt.show()


