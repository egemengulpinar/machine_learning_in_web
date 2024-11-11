########## Machine Learning In Web - Exercise 2, Practical(a) ##########
## Author : Hakki Egemen Gülpinar, Szymon Czajkowskis
## Date: 09.11.2024
## Subject: Signal Decomposition, Random Matrix Generation, PCA, ICA Reconstruction
#########################################################

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_time_series(t: np.ndarray, noise_scale: float, func) -> np.ndarray:
    """
    Generate time series with Gaussian noise per entry
    t: time variable
    noise_scale: scale of Gaussian noise
    func: function to generate the base time series
    return: time series with noise
    """
    noise = np.random.normal(0, noise_scale, len(t))
    f = func(t)
    return f + noise

def calculate_statistics(f: np.ndarray) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation for the given time series
    f: time series
    return: mean and standard deviation
    """
    return np.mean(f), np.std(f)


def draw_time_series(t: np.ndarray, series: list, xlabel: str, ylabel:str, title: str) -> None:
    """
    Draw the time series with mean and standard deviation
    t: time variable
    series: list of tuples containing (time series, mean, std, title, color)
    """
    plt.figure(figsize=(12, 8))
    
    for s, mean, std, label, color in series:
        plt.plot(t, s, label=label, color=color)
        plt.fill_between(t, mean - std, mean + std, color=color, alpha=0.08)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



####################### Signal decomposition ##################################
# Set the time variable 't' with a range of values 
t = np.arange(0, 80, 0.2) # values can be change for making different tests & analysis

# Define each time series with added Gaussian noise
f1 = generate_time_series(t, noise_scale=0.001, func=np.sin)                 # f1(t) = sin(t) + 0.001*η
f2 = generate_time_series(t, noise_scale=0.002, func=lambda t: 2 * (t % 1))  # f2(t) = 2*(t - floor(t)) + 0.002*η
f3 = generate_time_series(t, noise_scale=0.001, func=lambda t: 0.01 * t)     # f3(t) = 0.01*t + 0.001*η


# Calculate mean and standard deviation for each time series, analyze the statistical properties
f1_mean, f1_std = calculate_statistics(f1)
f2_mean, f2_std = calculate_statistics(f2)
f3_mean, f3_std = calculate_statistics(f3)

# Statistics for each time series
print("f1(t) - Mean:", f1_mean, "Std Dev:", f1_std)
print("f2(t) - Mean:", f2_mean, "Std Dev:", f2_std)
print("f3(t) - Mean:", f3_mean, "Std Dev:", f3_std)


series = [
    (f1, f1_mean, f1_std, "f1(t) = sin(t) + noise", "blue"),
    (f2, f2_mean, f2_std, "f2(t) = 2(t mod 1) + noise", "green"),
    (f3, f3_mean, f3_std, "f3(t) = 0.01t + noise", "red")
]

draw_time_series(t= t, series= series, xlabel="Time (t)", ylabel="Function value", title="Generated Time Series with Statistical Summary")

####################### 3x3 random matrix and x(t)=A⋅f(t)  Transformation ##################################

# Generate a random 3x3 matrix A for transformation
A = np.random.rand(3, 3) #created A matrix

# Stack the time series vertically to form a matrix for multiplication
F = np.vstack([f1, f2, f3]) #created f(t) matrix

transformed_series = A @ F #matrix multiplication, regarding to x(t)=A⋅f(t) 

# Extract each transformed series from the result
x1, x2, x3 = transformed_series

transformed_series = [
    (x1, *calculate_statistics(x1), "Transformed Series x1(t)", "purple"),
    (x2, *calculate_statistics(x2), "Transformed Series x2(t)", "orange"),
    (x3, *calculate_statistics(x3), "Transformed Series x3(t)", "cyan")
]

draw_time_series(t= t,series= transformed_series, xlabel="Time (t)", ylabel="Transformed Function Value", title="Transformed Time Series using Random Matrix")



