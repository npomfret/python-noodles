import numpy as np
import matplotlib.pyplot as plt


def as_time_series():
    samples = np.random.normal(size=1000)

    # Generate time index
    time_index = np.arange(len(samples))
    # Plot the samples as a time series
    plt.plot(time_index, samples)
    # Add title and axis labels
    plt.title('1000 Samples from the Standard Normal Distribution')
    plt.xlabel('Time')
    plt.ylabel('Value')
    # Display the plot
    plt.show()


def as_histogram():
    samples = np.random.normal(size=1000)

    plt.hist(samples, bins=30)
    # Add title and axis labels
    plt.title('Histogram of 1000 Samples from the Standard Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Display the plot
    plt.show()


def as_scatter_plot():
    # Generate normally distributed random values
    noise = np.random.normal(scale=0.2, size=1000)

    # Generate linearly increasing component
    trend = np.linspace(0, 1, 1000)

    # Combine noise and trend to generate samples
    samples = 0.5 + trend + noise

    # Plot the samples as a scatter plot
    time_index = range(1000)
    plt.scatter(time_index, samples)

    # Find the line of best fit
    slope, intercept = np.polyfit(time_index, samples, 1)
    line = slope * time_index + intercept

    # Plot the line of best fit
    plt.plot(line, color='red')

    # Add title and axis labels
    plt.title('1000 Samples with an Upward Trend')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')

    # Display the plot
    plt.show()


as_scatter_plot()
# as_time_series()
# as_histogram()