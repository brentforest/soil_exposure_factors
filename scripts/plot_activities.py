import numpy as np
import pandas as pd
from utilities_figs import *
from matplotlib.patches import PathPatch
from prep_grower_data import *

from matplotlib.ticker import ScalarFormatter
from pandas.api.types import CategoricalDtype

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/data/'
FIG_PATH = '../output/figures/activities/'
SHOW_FIGS = False

# Order in which activities are shown
ACTIVITY_ORDER = ['Preparing beds', 'Seeding', 'Transplanting', 'Weeding', 'Watering', 'Harvesting']
SEASON_ORDER = ['Spring', 'Summer', 'Fall', 'Winter']
#S_COLORS = ['darkorchid', 'limegreen', 'orange', 'royalblue']
#S_COLORS = ['mediumorchid', 'forestgreen', 'orange', 'royalblue']
#S_COLORS = ['darkorchid', 'greenyellow', 'chocolate', 'royalblue']
#S_COLORS = ['green', 'red', 'orange', 'blue']
PINK=(0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
MAGENTA=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
BLUE=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
GREEN=(0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
RED= (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
ORANGE=(1.0, 0.4980392156862745, 0.054901960784313725)

S_COLORS = [GREEN, ORANGE, RED, BLUE]


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    #TODO: UNUSED
    """

    ##iterating through Axes instances
    for ax in g.axes.flatten():

        ##iterating through axes artists:
        for c in ax.get_children():

            ##searching for PathPatches
            if isinstance(c, PathPatch):
                ##getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:,0])
                xmax = np.max(verts_sub[:,0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                ##setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:,0] == xmin,0] = xmin_new
                verts_sub[verts_sub[:,0] == xmax,0] = xmax_new

                ##setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin,xmax]):
                        l.set_xdata([xmin_new,xmax_new])


def plot_bar(df, x, y, hue, hue_order, filename, x_label, order='', y_label='Count', colors=S_COLORS, legend_cols=4):

    box_colors = ['w', 'w', 'w', 'w']

    # Remove snake case from hue
    #df.columns = df.columns.str.replace('_', ' ').str.capitalize()
    df[hue] = df[hue].str.replace('_', ' ').str.capitalize()

    # Set font
    font = {'family': 'Arial',
            'size': 5} # font must be between 5 and 7
    matplotlib.rc('font', **font)

    # Journal guidelines, single column: 86mm or 3.38583 inches wide
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.38583, 2.2), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.99)

    if order != '':
        sns.barplot(x=x, y=y, hue=hue, order=order, hue_order=hue_order, data=df, palette=colors)
    else:
        sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order, data=df, palette=colors)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Gridlines, horizontal and between items
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(axis='x', which='minor')
    ax.grid(axis='y', visible=True)
    ax.set_axisbelow(True)

    # Ticks and axis labels
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False  # ticks along the bottom edge are off
    )
    ax.tick_params(axis='x', which='major', pad=2, length=0)
    ax.tick_params(axis='y', which='major', pad=1, length=2)

    ax.yaxis.labelpad = 3.5

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=legend_cols)

    show_save_plot(show=SHOW_FIGS, filename=filename, path=FIG_PATH, transparent=True, format=['pdf', 'png'])


def plot_box_strip(df, x, y, order, filename, hue=None, hue_order=None, ymax=0, ylabel=''):

    box_colors = ['w', 'w', 'w', 'w']

    # Set font
    font = {'family': 'Arial',
            'size': 5} # font must be between 5 and 7
    matplotlib.rc('font', **font)

    # Journal guidelines, single column: 86mm or 3.38583 inches wide
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.38583, 2.5), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.99)

    if hue != None:
        ax = sns.boxplot(x=df[x], y=df[y], order=order, whis=0, hue=df[hue], hue_order=hue_order, showfliers=False, color='w', palette=box_colors,
                     medianprops=dict(color="black"))  # alternate setting: whis=np.inf

        """
        ax = sns.stripplot(x=x, y=y, hue=df[hue], order=order, hue_order=SEASON_ORDER,
                           dodge=True, jitter=0.3, size=2,
                           marker='o', clip_on=False, split=True, palette=s_colors)
        """

        # Strip plot; different marker shapes require multiple function calls
        df1 = df[(df['Season'] == 'Spring') | (df['Season'] == 'Fall')]
        ax = sns.stripplot(x=df1[x], y=df1[y], hue=df1[hue], order=order,
                           hue_order=hue_order, dodge=True, jitter=0.3, size=2, palette=S_COLORS,
                           marker='o', clip_on=False)

        df2 = df[(df['Season'] == 'Summer') | (df['Season'] == 'Winter')]
        ax = sns.stripplot(x=df2[x], y=df2[y], hue=df2[hue], order=order,
                           hue_order=hue_order, dodge=True, jitter=0.3, size=2, palette=S_COLORS,
                           marker='D', clip_on=False)

    else:

        ax = sns.boxplot(x=df[x], y=df[y], order=order, whis=0, showfliers=False, color='w',
                         medianprops=dict(color="black"))  # alternate setting: whis=np.inf

        ax = sns.stripplot(x=df[x], y=df[y], order=order,
                           dodge=True, jitter=0.3, size=2,
                           marker='o', clip_on=False, palette=['dimgrey'])

    #adjust_box_widths(fig, 0.8)

    # Gridlines, horizontal and between items
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.grid(axis='x', which='minor')
    ax.grid(axis='y', visible=True)
    ax.set_axisbelow(True)

    # Ticks and axis labels
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False  # ticks along the bottom edge are off
    )
    ax.tick_params(axis='x', which='major', pad=2, length=0)
    ax.tick_params(axis='y', which='major', pad=1, length=2)

    if ylabel !='':
        plt.ylabel(ylabel)

    if ymax >0:
        ax.set_ylim(top=ymax)
    ax.set_ylim(bottom=0)

    ax.yaxis.labelpad = 4.35
    if ymax >99:
        ax.yaxis.labelpad = 1.5

    plt.legend([], [], frameon=False)

    show_save_plot(show=SHOW_FIGS, filename=filename, path=FIG_PATH, transparent=True, format=['pdf', 'png'])


def tally_orientation(df):

    orient_cols = [col for col in df.columns if 'orient' in col]
    df = df[['grower_id', 'season', 'activity'] + orient_cols]

    # Convert from wide to long form
    df = df.melt(id_vars=['grower_id', 'season', 'activity'], var_name='orientation',
                     value_name='value')

    # Remove 'orient' from var
    df['orientation'] = df['orientation'].str.replace('orient_', '')

    # Compute counts
    # This sums over grower and season, so each count is a grower-season pair
    df = df.groupby(['activity', 'orientation'])['value'].sum().rename('count').reset_index()

    return(df)


def tally_soil_ingest(df, si_adds):

    # Tally number of respondents who consume each amount of soil, by season
    df = df.groupby(['season', 'soil_ingest_amt_code', 'soil_ingest_text', 'soil_ingest_fig'])['soil_ingest'].sum().rename('count').reset_index()

    # Zero growers ingested soil at the highest rates.
    # Manually add placeholder values for theses highest rates, otherwise they don't show up on the figure
    df = pd.concat([df, si_adds], sort=False)

    df = df.sort_values(['soil_ingest_amt_code'])

    return(df)


# Input ****************************************************************************************************************

si_adds = pd.read_csv(INPUT_PATH + 'soil_ingest_manual_adds.csv')

# **********************************************************************************************************************

# Note: Leave NaN values as Nan (e.g., don't replace them with zeroes) - these are respondents who did not engage in an activity

# Prepare grower data; this function is imported via separate script because it is also used by other scripts
(gro, sea, act) = prep_grower_data()

# Prep, plot soil ingestion
si = tally_soil_ingest(sea, si_adds)
si.to_csv(OUTPUT_PATH + 'soil_ingestion_summary.csv', index=False)
plot_bar(si, x='soil_ingest_fig', y='count', hue='season', hue_order=SEASON_ORDER, x_label='Amount ingested', filename='soil_ingestion_bar_chart')

# Prep, plot orientation
do = tally_orientation(act)
do.to_csv(OUTPUT_PATH + 'orientation_summary.csv', index=False)
o_order = ['Standing', 'Bending over', 'Kneeling', 'Sitting', 'Other']
plot_bar(do, x="activity", y="count", hue="orientation", hue_order=o_order, x_label='Activity', y_label='Number of grower-season pairs',
         order=ACTIVITY_ORDER, colors=[MAGENTA,  ORANGE, GREEN, BLUE, RED], legend_cols=5, filename='orientation_bar_chart')

# Activity strip plots *************************************************************************************************

# Remove snake case
act.columns= act.columns.str.replace('_', ' ').str.capitalize()

plot_box_strip(act, x='Activity', y='Gloves %', order=ACTIVITY_ORDER,
               ylabel='Percentage of time wearing gloves', filename='gloves_%', ymax=100)

plot_box_strip(act, x='Activity', y='Wash freq', order=ACTIVITY_ORDER,
               ylabel='Percentage of times hands were washed', filename='wash_freq', ymax=100)

plot_box_strip(act, x='Activity', y='Contact %', order=ACTIVITY_ORDER,
               ylabel='Percentage of time hands were in contact with soil', filename='contact_%', ymax=100)

plot_box_strip(act, x='Activity', y='Contact %', order=ACTIVITY_ORDER,
               ylabel='Percentage of time hands were in contact with soil', hue='Season', filename='contact_%_freq_by_season', ymax=100)

plot_box_strip(act, x='Activity', y='Hours per month by activity', order=ACTIVITY_ORDER,
               hue='Season', hue_order=SEASON_ORDER, ymax=200, filename='hours_per_month')

plot_box_strip(act, x='Activity', y='Hours per month by activity', order=ACTIVITY_ORDER,
               hue='Season', hue_order=SEASON_ORDER,filename='hours_per_month_no_max')

plot_box_strip(act, x='Activity', y='Hours per day by activity', order=ACTIVITY_ORDER,
               hue='Season', hue_order=SEASON_ORDER,filename='hours_per_day')

plot_box_strip(act, x='Activity', y='Days per month by activity', order=ACTIVITY_ORDER,
               hue='Season', hue_order=SEASON_ORDER,filename='days_per_month')
