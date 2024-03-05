import numpy as np
import pandas as pd
from utilities_stats import *
from utilities import *
from utilities_figs import *
from matplotlib.patches import PathPatch
import seaborn as sns
import matplotlib.pyplot as plt
from prep_grower_data import *


pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

# x range for exposure plots, in case we want these to be constant across methods or routes.
# set to "'" to use default values.
X_MIN = 0.0001
X_MAX = 0.007

# Note: I had previously converted soil ingestion rates to kg, but scientific notation makes it harder to interpret results tables.
# Instead I just divide the avg daily dose by 1,000,000.
CONCENTRATION = 400 # mg/kg soil
SI_DAILY = 378 # mg/day
SI_OUTDOOR = 45.25 # mg/hour
SI_INDOOR = 1.375 # mg/hour

N_SIMULATIONS = 5000
RANDOM_SEED = 5
SEASON_ORDER = ['Spring', 'Summer', 'Fall', 'Winter']
ACTIVITY_ORDER = ['Preparing beds', 'Seeding', 'Transplanting', 'Weeding', 'Watering', 'Harvesting']

PINK=(0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
MAGENTA=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
BLUE=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
GREEN=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
RED=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
ORANGE=(1.0, 0.4980392156862745, 0.054901960784313725)
SEASON_COLORS = [GREEN, ORANGE, RED, BLUE]
ACTIVITY_COLORS = ['sienna', 'gold', GREEN, ORANGE, BLUE, RED]

# Figure parameters
FITTED_LINE = False
FONT_SIZE = 7

# File paths
FIG_PATH = '../output/figures/exposure/'
OUTPUT_PATH = '../output/data/exposure/'
SHOW_FIGS = False

def prep_kwargs(k_dict):
#TODO: ADD THIS TO UTILITIES?
# Takes a dictionary of {variable name: variable} and packs non-blank values into a kwargs dictionary

    kwargs = {}
    for key, value in k_dict.items():
        if value != '':
            kwargs[key] = value # Add new key:value pair
    return kwargs


def plot_count(df, x, hue, order='',  file='fig', title=''):

    #kwargs = prep_kwargs({'row': row, 'col_order': col_order})
    #g = sns.FacetGrid(df, col=col, sharey=True, **kwargs)

    ax = sns.countplot(data=df, x=x, hue=hue, order=order, palette=['dimgrey', 'silver'])
    ax.set(title=title)
    show_save_plot(show=SHOW_FIGS, filename=file, path=FIG_PATH, transparent=True, format=['pdf','png'])


def plot_histo(df, x, file='fig'):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=80)
    #fig.subplots_adjust(left=0.13, right=0.99, bottom=0.08, top=0.96, wspace=.5, hspace=0.35)
    sns.histplot(ax=ax, data=df, x=x, kde=False, color='dimgray', edgecolor='none')
    show_save_plot(show=SHOW_FIGS, filename=file, path=FIG_PATH, transparent=True, format=['pdf', 'png'])


def plot_histo_subplots(df, x, row='', row_order='', sci_not=False, bins=50, binwidth='', log_scale=False,
               x_min = '', x_max='', y_max='',
               height_multiplier = 1.32, top=0.94, bottom=0.055,
               x_label='', y_label='', title='', file='fig'):

    # If log scale, we have to drop zeroes before plotting;
    # Flag alert if data include negative values (seaborn doesn't flag an error, and doesn't plot them - weird)
    if log_scale:

        if 0 in df[x].values:
            print('Dropping zeroes from log scale plot:')
            print(df[df[x]==0][['grower_id', 'season']])
            df = s_filter(df, col=x, excl_list=[0])

        if df[x].min() < 0:
            print('WARNING: negative values included in log scale plot; these will not be plotted')

    if row_order=='':
        row_order = df[row].drop_duplicates().tolist()
    nrows = len(row_order)

    # Create subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(2.8, nrows*height_multiplier), dpi=80, sharex=False, sharey=True)
    fig.subplots_adjust(left=0.14, right=0.95, bottom=bottom, top=top, wspace=.5, hspace=0.46)

    # Loop through subplots
    n = 0
    for r in row_order:

        dfa = df[df[row] == r]

        # Plot
        kwargs = prep_kwargs({'bins': bins, 'binwidth': binwidth})
        sns.histplot(ax=axs[n], data=dfa, x=x, kde=False, color='dimgray', edgecolor='none', log_scale=log_scale, **kwargs)

        # Titles and labels
        # Note there are no data for some grower-season pairs
        axs[n].set_title(str(r) + ' (N=' + str(len(dfa[row])) + ')', fontsize=FONT_SIZE, pad=3)
        axs[n].set_xlabel('')
        axs[n].set_ylabel('')
        axs[n].tick_params(axis='x', which='major', pad=1.1)
        #axs[n].tick_params(axis='y', which='major', pad=1.5) # Distance between ticks and ticklabels

        # Adjust axis range
        if x_min!='':
            axs[n].set_xlim(x_min, x_max)
        if y_max!='':
            axs[n].set_ylim(0, y_max)

        # Vertical lines for median, 5th and 95th quantiles
        axs[n].axvline(x=dfa[x].median(), color='r')
        axs[n].axvline(x=dfa[x].quantile(q=0.05), color='r', linestyle=':')
        axs[n].axvline(x=dfa[x].quantile(q=0.95), color='r', linestyle=':')

        n += 1

    # Main title
    fig.suptitle(title)

    # Create a big plot, with no grid or tick labels, so we can use a common y-axis label
    # https://stackoverflow.com/questions/6963035/how-to-set-common-axes-labels-for-subplots
    fig.add_subplot(111, frameon=False)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    # Scientific notation
    if (sci_not == True) & (log_scale == False):
        plt.ticklabel_format(axis='x', style="sci", scilimits=(0, 0))

    show_save_plot(show=SHOW_FIGS, filename=file, path=FIG_PATH, transparent=True, format=['pdf'])


def plot_stacked_bar(df, order, file, colors='', title=''):

    # Sort by total annual exposure
    df['total'] = df[order].sum(axis=1)
    df = df.sort_values(by='total')

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.8, 2.8), dpi=80)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.12, top=0.94, hspace=0.12)
    bplot = df[['grower_id'] + order].plot(ax=ax, kind='bar', x='grower_id', stacked=True, width=0.9, color=colors)

    # Compute % of total exposure
    """
    for s in stack_order:
        df[s] = df[s] / df['total'] * 100

    # Plot % exposure
    bplot = df[['grower_id'] + stack_order].plot(ax=axs[1], kind='bar', x='grower_id', stacked=True, width=0.9, **kwargs)
    """

    fig.suptitle(title)

    # Axis
    ax.set_ylim([0, 1.1])
    #ax.get_xaxis().set_ticks([])
    ax.set_ylabel('mg/kg BW per year')
    ax.set_xlabel('Grower ID')
    ax.get_legend().remove()
    #ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.grid(axis='y')
    ax.set_axisbelow(True)

    # Reverse order of legend so it matches how they are displayed
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=FONT_SIZE)

    show_save_plot(show=SHOW_FIGS, filename=file, path=FIG_PATH, transparent=True, format='pdf')


# Input ****************************************************************************************************************

# TODO: RUN EXPOSURE_ASSESSMENT FIRST!

emp_sa = pd.read_csv(OUTPUT_PATH + 'empirical_exposure_by_season_activity_method_3_outdoor_only.csv')
sim_sa = pd.read_csv(OUTPUT_PATH + 'simulated_exposure_by_season_activity_method_3_outdoor_only.csv')

emp_s = pd.read_csv(OUTPUT_PATH + 'empirical_exposure_by_season.csv')
sim_s = pd.read_csv(OUTPUT_PATH + 'simulated_exposure_by_season.csv')

emp_annual_s = pd.read_csv(OUTPUT_PATH + 'empirical_exposure_annual_by_season.csv')
emp_annual_a = pd.read_csv(OUTPUT_PATH + 'empirical_exposure_annual_by_activity.csv')
# **********************************************************************************************************************

# Set font
font = {'family': 'Arial',
        'size': FONT_SIZE}
matplotlib.rc('font', **font)

# TODO: needs updating, not essential
"""
# Plot activity histograms
for a in act['activity'].unique():
    plot_histo_subplots(act[act['activity'] == a], x='hours_per_month', row='season', row_order=SEASON_ORDER,
                    file='activity_freq/' + a, binwidth=6, sci_not=False)

# Plot body weight based on age, sex
plot_histo_subplots(gro, x='body_weight_kg', row='sex', file='bw_efh', sci_not=False)
"""

print('\nPlotting annual bar charts:') # *******************************************************************************

# by season
for m in ['method_1', 'method_2', 'method_3']:
    title = m.replace('_',' ').capitalize() + ' by season'
    df = emp_annual_s.pivot(index='grower_id', columns='season', values='avg_daily_dose_' + m).reset_index()
    plot_stacked_bar(df, order=SEASON_ORDER, colors=SEASON_COLORS, file='exposure_annual_by_season_empirical_' + m, title=title)

# by indoor/outdoor
for m in ['method_2', 'method_3']:
    title = m.replace('_',' ').capitalize() + ' by indoor/outdoor'
    df = emp_annual_s.groupby('grower_id').sum().reset_index() \
        .rename(columns={'avg_daily_dose_' + m + '_outdoor': 'outdoor',
                       'avg_daily_dose_' + m + '_indoor': 'indoor'})
    order = ['outdoor', 'indoor']
    plot_stacked_bar(df, order=order, colors=[GREEN, ORANGE], file='exposure_annual_indoor_outdoor_empirical_' + m, title=title)

# by activity
df = emp_annual_a.pivot(index='grower_id', columns='activity', values='avg_daily_dose_method_3_outdoor').reset_index()
plot_stacked_bar(df, order=ACTIVITY_ORDER, colors=ACTIVITY_COLORS, file='exposure_annual_by_activity_empirical_method_3', title='Method 3 by activity, outdoor only')

print('\nPlotting by season:') # ***************************************************************************************

kwargs = {'log_scale': True, 'x_label': 'Average daily dose (mg/kg BW per day)'}

for m in ['method_1', 'method_2', 'method_3']:

    print('\nEmpirical,', m)
    plot_histo_subplots(df=emp_s, x='avg_daily_dose_' + m, row='season', row_order=SEASON_ORDER,
                            bottom=0.075, top=0.92, binwidth=0.06, x_min=X_MIN, x_max=X_MAX, y_max=15,
                            y_label='Number of growers', file='exposure_by_season_empirical_' + m,
                            title=m.replace('_', ' ').capitalize(), **kwargs)

    print('\nSimulated,', m)
    plot_histo_subplots(df=sim_s, x='avg_daily_dose_' + m, row='season', row_order=SEASON_ORDER,
                            bottom = 0.075, top = 0.92, binwidth=0.03, x_min=X_MIN, x_max=X_MAX, y_max=470,
                            y_label='Number of growers', file='exposure_by_season_simulated_' + m,
                            title=m.replace('_', ' ').capitalize(), **kwargs)

print('\nPlotting by season and activity, method 3 only:') # ***********************************************************
x_min = 0.000001
x_max = 0.004
x = 'avg_daily_dose_method_3_outdoor'
for s in SEASON_ORDER:

    print('\nEmpirical,', m)
    emp = s_filter(emp_sa, col='season', list=[s])
    plot_histo_subplots(df=emp, x=x, row='activity', row_order=ACTIVITY_ORDER,
                        bottom=0.075, top=0.92, binwidth = 0.15, x_min=x_min, x_max=x_max, y_max=8.8,
                        y_label='Number of growers', file='exposure_by_season_activity_empirical_' + s,
                        title=s, **kwargs)

    print('\nSimulated,', m)
    sim = s_filter(sim_sa, col='season', list=[s])
    plot_histo_subplots(df=sim, x=x, row='activity', row_order=ACTIVITY_ORDER,
                        bottom=0.075, top=0.92, binwidth=0.03, x_min=x_min, x_max=x_max,y_max=200,
                        y_label='Number of growers', file='exposure_by_season_activity_simulated_' + s,
                        title=s, **kwargs)
