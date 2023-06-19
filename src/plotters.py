from math import log
import seaborn as sns
import numpy as np
from matplotlib import rc_context
import matplotlib.pyplot as plt
from src.helpers import smooth


def custom_line_plots(ids_start, ids_stop, groups, data, x, y, hue,
              fs=18, lw=5, x_label=None, y_label=None, k=1, aspect_ratio=0.5,
              weight=None, ylim=(10, 105), palette='bright', reverse=True):
    """
    TODO
    """
    nrows = len(ids_start)
    with rc_context({'lines.linewidth': lw * k, 'font.size': fs * k}):
        fig, axs = plt.subplots(nrows=nrows,
                               figsize=(k * 16, k * 16 * aspect_ratio * nrows))
        sns.set_style("darkgrid")
        for (ax, id_start, id_stop) in zip(axs, ids_start, ids_stop):
            data2 = data[data[hue].isin(groups[id_start:id_stop])]
            if weight is not None:
              for source in groups[id_start:id_stop]:
                data2.loc[data2[hue]==source, y] = smooth(data2.loc[data2[hue]==source, y].tolist(), weight=weight,
                                                          reverse=reverse)
            sns.lineplot(ax=ax, data=data2, x=x, y=y, hue=hue, palette=palette)
            if x_label is not None: ax.set_xlabel(x_label)
            if y_label is not None: ax.set_ylabel(y_label)
            ax.set_ylim(*ylim)
        fig.tight_layout()
        plt.show()



def custom_bar_plot(data, x, y, hue, fs=18, k=1, aspect_ratio=0.5, x_rotation=45,
                    x_vals=None, show_values=False, ylim=(0, 100)):
    """
    TODO
    """
    with rc_context({'font.size': fs * k}):
        plt.figure(figsize=(k * 16, k * 16 * aspect_ratio))
        sns.set_style("darkgrid")
        if x_vals is not None:
          plot = sns.barplot(data=data[data[x].isin(x_vals)], x=x, y=y, hue=hue)
        else:
          plot = sns.barplot(data=data, x=x, y=y, hue=hue)
        if show_values:
          for container in plot.containers:
            plot.bar_label(container, fmt='%.0f')
        plt.ylim(*ylim)
        plt.xticks(rotation=x_rotation)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.show()


def pie_plots(data_pos, data_neg, tokens_col, freq_col, fs1=18, fs2=5, k=1, aspect_ratio=0.5,
              top=None, topics=None, subplots_adjust={}, pads=[10, 50], groups=None):
    """
    TODO
    """
    nrows = 2
    ncols = len(data_pos)
    if groups is None:
        groups = data_pos.index.tolist()
    if True:
        fig, axs = plt.subplots(nrows, ncols, figsize=(k * 16 * nrows, k * 16 * aspect_ratio))
        for i, data in enumerate([data_pos, data_neg]):
            for it, ax in enumerate(axs[i, :]):
                tokens = data.iloc[it][tokens_col]
                freq = data.iloc[it][freq_col]
                if top is not None:
                    tokens = tokens[:top]
                    freq = freq[:top]
                colors = plt.get_cmap('jet')(np.linspace(0, 1, len(tokens)))[::-1, :]
                ax.pie(freq, labels=tokens, colors=colors, autopct='%.0f%%', textprops={'fontsize': fs2 * k})

        if topics is not None:
            for ax, col in zip(axs[0, :], groups):
                ax.set_title(col, fontsize=fs1 * k, pad=pads[0])
        for ax, row in zip(axs[:, 0], topics):
            ax.set_ylabel(row, fontsize=fs1 * k, labelpad=pads[1])
        fig.tight_layout()
        fig.subplots_adjust(**subplots_adjust)
        plt.show()


def wrapper_for_pie_plots(ids_start, ids_end, data_pos, data_neg, tokens_col, freq_col,
                          fs1, fs2, k, aspect_ratio, top, topics, pads, subplots_adjust):
    """
    TODO
    """
    for (start, end) in zip(ids_start, ids_end):
        pie_plots(data_pos=data_pos.iloc[start:end],
                  data_neg=data_neg.iloc[start:end],
                  tokens_col=tokens_col,
                  freq_col=freq_col,
                  fs1=fs1,
                  fs2=fs2,
                  k=k,
                  aspect_ratio=aspect_ratio,
                  top=top,
                  topics=topics,
                  pads=pads,
                  subplots_adjust=subplots_adjust)


def line_plot(x, y, fs=18, lw=5, x_label=None, y_label=None, k=1, aspect_ratio=0.5, multiple=False, labels=None, kwargs={},
              x_scale=None, y_scale=None):
    """
    TODO
    """
    with rc_context({'lines.linewidth': lw * k, 'font.size': fs * k}):
        plt.figure(figsize=(k * 16, k * 16 * aspect_ratio))
        sns.set_style("darkgrid")
        if multiple:
            for it in range(len(x)):
                sns.lineplot(x=x[it], y=y[it], label=labels[it], **kwargs)
        else:
            sns.lineplot(x=x, y=y, label=labels[0], **kwargs)
        if x_label is not None: plt.xlabel(x_label)
        if y_label is not None: plt.ylabel(y_label)
        if x_scale is not None: plt.xscale(x_scale)
        if y_scale is not None: plt.yscale(y_scale)
        plt.legend()
        plt.show()


def scatter_plot(x, y, fs=18, lw=5, x_label=None, y_label=None, k=1, aspect_ratio=0.5, multiple=False, labels=None, kwargs={},
                 x_scale=None, y_scale=None):
    """
    TODO
    """
    with rc_context({'lines.linewidth': lw * k, 'font.size': fs * k}):
        plt.figure(figsize=(k * 16, k * 16 * aspect_ratio))
        sns.set_style("darkgrid")
        if multiple:
            for it in range(len(x)):
                sns.scatterplot(x=x[it], y=y[it], label=labels[it], **kwargs)
        else:
            sns.scatterplot(x=x, y=y, label=labels[0], **kwargs)
        if x_label is not None: plt.xlabel(x_label)
        if y_label is not None: plt.ylabel(y_label)
        if x_scale is not None: plt.xscale(x_scale)
        if y_scale is not None: plt.yscale(y_scale)
        plt.legend()
        plt.show()

def hist_plot(x, fs=18, lw=5, x_label=None, k=1, aspect_ratio=0.5, multiple=False, labels=None, kwargs={}):
    """
    TODO
    """
    with rc_context({'lines.linewidth': lw * k, 'font.size': fs * k}):
        plt.figure(figsize=(k * 16, k * 16 * aspect_ratio))
        sns.set_style("darkgrid")
        if multiple:
            for it, item in enumerate(x):
                sns.histplot(item, bins=int(1+3.32*log(len(item))), label=labels[it], **kwargs)
        else:
            sns.histplot(x, bins=int(1+3.32*log(len(x))), label=labels[0], **kwargs)
        if x_label is not None: plt.xlabel(x_label)
        plt.legend()
        plt.show()


def kde_plot(x, fs=18, lw=5, x_label=None, k=1, aspect_ratio=0.5, multiple=False, labels=None, kwargs={}):
    """
    TODO
    """
    with rc_context({'lines.linewidth': lw * k, 'font.size': fs * k}):
        plt.figure(figsize=(k * 16, k * 16 * aspect_ratio))
        sns.set_style("darkgrid")
        if multiple:
            for it, item in enumerate(x):
                sns.kdeplot(item, fill=True, label=labels[it], **kwargs)
        else:
            sns.kdeplot(x, fill=True, label=labels[0], **kwargs)
        if x_label is not None: plt.xlabel(x_label)
        plt.legend()
        plt.show()