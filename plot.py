import numpy as np
import matplotlib.pyplot as plt


def smoothed_plot(file, data, x_label="Timesteps", x=None, window=5):
    N = len(data)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(data[max(0, t - window):(t + 1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Success rate')
    plt.xlabel(x_label)
    plt.plot(x, running_avg)
    plt.savefig(file)
    plt.close()
