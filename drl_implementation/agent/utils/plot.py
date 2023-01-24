import json
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy as dcp


def smoothed_plot(file, data, x_label="Timesteps", y_label="Success rate", window=5):
    N = len(data)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(data[max(0, t - window):(t + 1)])
    x = [i for i in range(N)]
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if x_label == "Epoch":
        x_tick_interval = len(data) // 10
        plt.xticks([n * x_tick_interval for n in range(11)])
    plt.plot(x, running_avg)
    plt.savefig(file, bbox_inches='tight', dpi=500)
    plt.close()


def smoothed_plot_multi_line(file, data,
                             legend=None, legend_loc="upper right",
                             x_label='Timesteps', y_label="Success rate", window=5):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if x_label == "Epoch":
        x_tick_interval = len(data[0]) // 10
        plt.xticks([n * x_tick_interval for n in range(11)])

    for t in range(len(data)):
        N = len(data[t])
        x = [i for i in range(N)]
        if window != 0:
            running_avg = np.empty(N)
            for n in range(N):
                running_avg[n] = np.mean(data[t][max(0, n - window):(n + 1)])
        else:
            running_avg = data[t]

        plt.plot(x, running_avg)

    if legend is None:
        legend = [str(n) for n in range(len(data))]
    plt.legend(legend, loc=legend_loc)
    plt.savefig(file, bbox_inches='tight', dpi=500)
    plt.close()


def smoothed_plot_mean_deviation(file, data_dict_list, title=None,
                                 vertical_lines=None, horizontal_lines=None, linestyle='--', linewidth=4,
                                 x_label='Timesteps', x_axis_off=False,
                                 y_label="Success rate", window=5, ylim=(None, None), y_axis_off=False,
                                 legend=None, legend_only=False, legend_file=None, legend_loc="upper right",
                                 legend_title=None, legend_bbox_to_anchor=None, legend_ncol=4, legend_frame=False,
                                 handlelength=2):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    if not isinstance(data_dict_list, list):
        data_dict_list = [data_dict_list]

    if y_axis_off:
        plt.ylabel(None)
        plt.yticks([])
    else:
        plt.ylabel(y_label)
    if ylim[0] is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title)

    if x_axis_off:
        plt.xlabel(None)
        plt.xticks([])
    else:
        plt.xlabel(x_label)
        if x_label == "Epoch":
            x_tick_interval = len(data_dict_list[0]["mean"]) // 10
            plt.xticks([n * x_tick_interval for n in range(11)])

    handles = [Line2D([0], [0], color=colors[i], linewidth=linewidth) for i in range(len(data_dict_list))]
    if legend is not None:
        legend_plot = plt.legend(handles, legend, handlelength=handlelength,
                                 title=legend_title, loc=legend_loc, labelspacing=0.15,
                                 bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, frameon=legend_frame)
        if legend_only:
            assert legend_file is not None, 'specify legend save path'
            fig = legend_plot.figure
            fig.canvas.draw()
            bbox = legend_plot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(legend_file, dpi=500, bbox_inches=bbox)
            plt.close()
            return

    N = len(data_dict_list[0]["mean"])
    x = [i for i in range(N)]
    for i in range(len(data_dict_list)):
        case_data = data_dict_list[i]
        for key in case_data:
            running_avg = np.empty(N)
            for n in range(N):
                running_avg[n] = np.mean(case_data[key][max(0, n - window):(n + 1)])

            case_data[key] = dcp(running_avg)

        plt.fill_between(x, case_data["upper"], case_data["lower"], alpha=0.3, color=colors[i], label='_nolegend_')
        plt.plot(x, case_data["mean"], color=colors[i])

    if horizontal_lines is not None:
        for n in range(len(horizontal_lines)):
            plt.axhline(y=horizontal_lines[n], color=colors[len(data_dict_list) + n], xmin=0.05, xmax=0.95,
                        linestyle=linestyle, linewidth=linewidth)
    if vertical_lines is not None:
        assert horizontal_lines is None
        for n in range(len(vertical_lines)):
            plt.axvline(x=vertical_lines[n], color=colors[len(data_dict_list) + n], ymin=0.05, ymax=0.95,
                        linestyle=linestyle, linewidth=linewidth)

    plt.savefig(file, bbox_inches='tight', dpi=500)
    plt.close()


def get_mean_and_deviation(data, save_data=False, file_name=None):
    upper = np.max(data, axis=0)
    lower = np.min(data, axis=0)
    mean = np.mean(data, axis=0)
    statistics = {"mean": mean.tolist(),
                  "upper": upper.tolist(),
                  "lower": lower.tolist()}
    if save_data:
        assert file_name is not None
        json.dump(statistics, open(file_name, 'w'))
    return statistics
