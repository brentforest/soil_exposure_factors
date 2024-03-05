import numpy as np
import pandas as pd
from utilities import *
from prep_grower_data import *

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

INPUT_PATH = '../output/data/'
OUTPUT_PATH = '../output/data/summary_tables/'
SEASON_ORDER = ['Spring', 'Summer', 'Fall', 'Winter']

def summarize(df, attributes, groupby=''):
# Generate table of summary statistics

    # If groupby is blank, create a helper column
    if groupby=='':
        df['helper_col'] = 'helper_col'
        groupby=['helper_col']

    # Wide to long form
    df = df.melt(id_vars=groupby, var_name='attribute', value_name='value')

    # Filter
    df = s_filter(df, col='attribute', list=attributes)
    df['value'] = df['value'].astype(float)

    df = df.groupby(groupby + ['attribute']).describe(percentiles=[.05, .25, .5, .75, .95], include='all').reset_index()

    # df has two header rows; flatten to just one and rename columns
    # https://towardsdatascience.com/how-to-flatten-multiindex-columns-and-rows-in-pandas-f5406c50e569
    df.columns = [''.join(col) for col in df.columns.values]
    df.columns = df.columns.str.replace('value','')
    df = df.rename(columns={'count': 'n'})

    # Sort
    #df = s_categorical_sort(df, col='season', sort_order=SEASON_ORDER)
    df = df.sort_values(by='attribute')

    if groupby==['helper_col']:
        df = df.drop(columns='helper_col')

    return df


# Input ****************************************************************************************************************

# Questionnaire data (prepped inputs)
gd_g = pd.read_csv(INPUT_PATH + 'grower_data/grower_data_by_grower.csv')
gd_s = pd.read_csv(INPUT_PATH + 'grower_data/grower_data_by_season.csv')
gd_sa = pd.read_csv(INPUT_PATH + 'grower_data/grower_data_by_season_activity.csv')

# Exposure estimates, also includes exposure factors
emp_s = pd.read_csv(INPUT_PATH + 'exposure/empirical_exposure_by_season.csv')
sim_s = pd.read_csv(INPUT_PATH + 'exposure/simulated_exposure_by_season.csv')

# Method 3 only, broken out by season and activity
emp_sa = pd.read_csv(INPUT_PATH + 'exposure/empirical_exposure_by_season_activity_method_3_outdoor_only.csv')
sim_sa = pd.read_csv(INPUT_PATH + 'exposure/simulated_exposure_by_season_activity_method_3_outdoor_only.csv')

season_order = pd.DataFrame([['All seasons', 0], ['Spring', 1], ['Summer', 2], ['Fall', 3], ['Winter', 4]],
                            columns=['season', 'season_order'])

# Normality tests
#norm = pd.read_csv(DATA_PATH + 'stat_tests/normality_tests.csv')

# **********************************************************************************************************************

# Summarize exposure factors

# by grower
sum_g = summarize(gd_g, ['body_weight_kg', 'farm_size_mean_ha'])
sum_g[['specificity', 'season', 'activity']] = ['by grower', 'n/a', 'n/a']

# by season
sum_s = pd.concat([
    summarize(gd_s, ['farm_size_by_season_ha', 'soil_ingest_days_per_month', 'soil_face_days_per_month', 'produce_consume_freq', 'consume_snacks_freq'], groupby=['season']),
    summarize(emp_s, ['days_per_month_at_site', 'hours_per_month_at_site',
                      'hours_per_month_all_activities'], groupby=['season'])
], sort=False)
sum_s[['specificity', 'activity']] = ['by grower, season', 'n/a']

# by season and activity
sum_sa = summarize(gd_sa, ['gloves_%', 'wash_freq', 'contact_%',
                      'hours_per_day_by_activity', 'days_per_month_by_activity',
                      'hours_per_month_by_activity'], groupby=['season', 'activity'])
sum_sa['specificity'] = 'by grower, season, activity'

# combine into one file
pd.concat([sum_g, sum_s, sum_sa], sort=False) \
    .merge(season_order, on='season', how='left') \
    .sort_values(by=['specificity', 'attribute', 'season_order', 'activity']) \
    .to_csv(OUTPUT_PATH + 'summary_survey_data.csv', index=False)

"""
# Merge with normality tests (OLD CODE)
norm = norm.rename(columns={'y':'attribute'})
norm = norm[['season', 'activity', 'attribute', 'normal', 'log_normal']]
ef = s_merge(ef, norm, on=['season', 'activity', 'attribute'], how='left', validate='1:1')
"""

# **********************************************************************************************************************

# Summarize exposure results
attributes = ['ef_method_1', 'avg_daily_dose_method_1',
              'ef_method_2_indoor', 'ef_method_2_outdoor', 'avg_daily_dose_method_2_indoor', 'avg_daily_dose_method_2_outdoor', 'avg_daily_dose_method_2',
              'ef_method_3_indoor', 'ef_method_3_outdoor', 'avg_daily_dose_method_3_indoor', 'avg_daily_dose_method_3_outdoor', 'avg_daily_dose_method_3']
"""
attributes = ['avg_daily_dose_method_1',
              'avg_daily_dose_method_2_indoor', 'avg_daily_dose_method_2_outdoor', 'avg_daily_dose_method_2',
              'avg_daily_dose_method_3_indoor', 'avg_daily_dose_method_3_outdoor', 'avg_daily_dose_method_3']
attributes = ['avg_daily_dose_method_1', 'avg_daily_dose_method_2', 'avg_daily_dose_method_3']
"""

# by season
groupby = ['season']
emp_s = summarize(emp_s, attributes, groupby)
emp_s[['specificity', 'grower_data', 'activity']] = ['by grower, season', 'empirical', 'n/a']
sim_s = summarize(sim_s, attributes, groupby)
sim_s[['specificity', 'grower_data', 'activity']] = ['by grower, season', 'simulated', 'n/a']

# by season and activity
attributes = ['avg_daily_dose_method_3_outdoor']
groupby = ['season', 'activity']
emp_sa = summarize(emp_sa, attributes, groupby)
emp_sa[['specificity', 'grower_data']] = ['by grower, season, activity', 'empirical']
sim_sa = summarize(sim_sa, attributes, groupby)
sim_sa[['specificity', 'grower_data']] = ['by grower, season, activity', 'simulated']

# combine into one file
pd.concat([emp_s, emp_sa, sim_s, sim_sa], sort=False) \
    .merge(season_order, on='season', how='left') \
    .sort_values(by=['grower_data', 'specificity', 'attribute', 'season_order', 'activity']) \
    .to_csv(OUTPUT_PATH + 'summary_exposure_results.csv', index=False)