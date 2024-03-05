import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
import math
from matplotlib.ticker import ScalarFormatter
from utilities_stats import *
from pandas.api.types import CategoricalDtype

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None


def plot_error_bars(ax, df, x_var, y_var, down_var, up_var):
# Input: long-form dataframe with columns for x var, y var, and up/down y offset for each error bar

    # 2D array: 1st row = down distance, 2nd row = up distance
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
    y_offset = df[[down_var, up_var]].values.transpose()

    # Draw error bars
    ax.errorbar(x=df[x_var], y=df[y_var], yerr=y_offset,
                fmt='none', ecolor='dimgrey', elinewidth=0.75, capsize=1.5, capthick=0.75)


def rotate_x_labels(ax, degrees=45, x_shift=4.2, y_shift=1):
# Rotate xtick labels, shift position
# Be sure to call this function AFTER other functions that may reset xtick label positions,
# otherwise the repositioning may be undone

    ax.set_xticklabels(ax.get_xticklabels(), rotation=degrees, ha='right')

    # Fine tuned positioning:
    # https://stackoverflow.com/questions/48326151/moving-matplotlib-xticklabels-by-pixel-value/48326438#48326438
    trans = mtrans.Affine2D().translate(x_shift, y_shift)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform() + trans)


def show_save_plot(show=True, filename='', path='', transparent=False, format=['png']):
# Show or save figure, then close

    #if show==None:
    #    return
    if show:
        plt.show()
    else:
        if type(format) == str:
            format = [format]
        for f in format:
            if filename != '':
                if f == 'png':
                    plt.savefig(path + filename + '.png', format='png', dpi=300, transparent=transparent)
                elif f == 'pdf':
                            plt.savefig(path + filename + '.pdf', format='pdf', transparent=transparent)
                else:
                    print('ALERT: unrecognized image format')

        # Wipe figure after save
        plt.close()