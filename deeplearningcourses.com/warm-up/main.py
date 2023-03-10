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


def make_scatter_plot_with_trend():
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


def generate_trend_and_show_cumulative_sum():
    # Generate normally distributed random values
    #  need a large scale here else we don't get a 'nice' plot
    noise = np.random.normal(scale=20, size=1000)

    # Generate linearly increasing component
    trend = np.linspace(0, 1, 1000)

    # Combine noise and trend to generate samples
    samples = (0.5 + trend + noise) - trend.mean()

    # Plot the samples as a scatter plot
    plt.scatter(range(1000), samples)

    # Find the line of best fit
    slope, intercept = np.polyfit(range(1000), samples, 1)
    line = slope * np.arange(1000) + intercept

    # Plot the line of best fit
    plt.plot(line, color='red')

    # Add title and axis labels
    plt.title('1000 Samples with an Upward Trend and a Mean of Zero')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')

    # Display the plot
    plt.show()

    # Calculate the cumulative sum of the samples
    cumulative_sum = np.cumsum(samples)

    # Plot the cumulative sum
    plt.plot(cumulative_sum)

    # Add title and axis labels
    plt.title('Cumulative Sum of 1000 Samples with an Upward Trend and a Mean of Zero')
    plt.xlabel('Sample Index')
    plt.ylabel('Cumulative Sum')

    # Display the plot
    plt.show()


def exercise_4():
    global mean, cov
    mean = [0, 0]
    cov = [[1, -0.5], [-0.5, 2]]
    # Generate 1000 samples from multivariate normal distribution
    samples = np.random.multivariate_normal(mean, cov, 1000)
    # Plot samples
    plt.scatter(samples[:, 0], samples[:, 1], s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('1000 Samples from a Multivariate Normal Distribution')
    plt.show()


exercise_4()

# generate_trend_and_show_cumulative_sum()
# make_scatter_plot_with_trend()
# as_time_series()
# as_histogram()