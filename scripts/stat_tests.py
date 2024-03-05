import numpy as np
import pandas as pd
import math
from utilities_stats import *
from utilities import *
import scipy.stats as stats
import statsmodels.api as sm
from utilities import *
from utilities_stats import *

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

INPUT_PATH = '../output/data/'
OUTPUT_PATH = '../output/data/stat_tests/'

def test_kruskal_wallace_wrapper(gd_s, gd_sa):

   # by season
    groups_tuple = (('season', 'Spring'), ('season', 'Summer'), ('season', 'Fall'), ('season', 'Winter'))
    kw_s = pd.DataFrame()
    for y in ['days_per_month_at_site', 'hours_per_month_at_site']:
        kw_s = pd.concat([kw_s, test_kruskal_wallace(gd_s, y=y, groups=groups_tuple)], sort=False)

    # need to group first before doing kruskall wallace since the season-activity data are clustered
    hours_act = gd_sa.groupby(['grower_id', 'season'])['hours_per_month_by_activity'].sum().rename(
        'hours_per_month_all_activities').reset_index()
    for y in ['hours_per_month_all_activities']:
        kw_s = pd.concat([kw_s, test_kruskal_wallace(hours_act, y=y, groups=groups_tuple)], sort=False)

    kw_s['specificity'] = 'by season'

    #kw.to_csv(OUTPUT_PATH + 'kruskall_wallace_by_season.csv', index=False)

    # by activity
    groups_tuple = (
    ('activity', 'Harvesting'), ('activity', 'Weeding'), ('activity', 'Preparing beds'), ('activity', 'Watering'),
    ('activity', 'Transplanting'), ('activity', 'Seeding'))
    kw_a = pd.DataFrame()

    # need to group first before doing kruskall wallace since the season-activity data are clustered
    hours_act = gd_sa.groupby(['grower_id', 'activity'])['hours_per_month_by_activity'].sum().rename(
        'hours_per_month_all_seasons').reset_index()
    for y in ['hours_per_month_all_seasons']:
        kw_a = pd.concat([kw_a, test_kruskal_wallace(hours_act, y=y, groups=groups_tuple)], sort=False)

    kw_a['specificity'] = 'by activity'

    pd.concat([kw_s, kw_a]).to_csv(OUTPUT_PATH + 'kruskall_wallace.csv', index=False)


def test_normality_log(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, log):

    gd_norm = pd.concat([
        test_normality(gd, y='body_weight_kg', log=log),
        test_normality(gd_s, y='days_per_month_at_site', groupby='season', log=log),
        test_normality(gd_s, y='hours_per_month_at_site', groupby='season', log=log)
    ], sort=False)
    emp_norm = pd.concat([
        test_normality(emp_s, y='avg_daily_dose_method_1', groupby='season', log=log),
        test_normality(emp_s, y='avg_daily_dose_method_2', groupby='season', log=log),
        test_normality(emp_s, y='avg_daily_dose_method_3', groupby='season', log=log),
        test_normality(emp_sa, y='hours_per_month_by_activity', groupby=['season', 'activity'], log=log),
        test_normality(emp_sa, y='avg_daily_dose_method_3_outdoor', groupby=['season', 'activity'], log=log)
    ], sort=False)
    sim_norm = pd.concat([
        test_normality(sim_s, y='avg_daily_dose_method_1', groupby='season', log=log),
        test_normality(sim_s, y='avg_daily_dose_method_2', groupby='season', log=log),
        test_normality(sim_s, y='avg_daily_dose_method_3', groupby='season', log=log),
        test_normality(sim_sa, y='hours_per_month_by_activity', groupby=['season', 'activity'], log=log),
        test_normality(sim_sa, y='avg_daily_dose_method_3_outdoor', groupby=['season', 'activity'], log=log)
    ], sort=False)

    gd_norm['grower_data'] = '0n/a' # the "0" helps with sorting
    emp_norm['grower_data'] = 'empirical'
    sim_norm['grower_data'] = 'simulated'

    return pd.concat([gd_norm, emp_norm, sim_norm], sort=False)


def test_normality_wrapper(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, season_order):

    # Drop "non-doers"
    emp_s = emp_s[emp_s['hours_per_month_at_site'] != 0]
    emp_sa = emp_sa[emp_sa['hours_per_month_by_activity'] != 0]
    sim_s = sim_s[sim_s['hours_per_month_at_site'] != 0]
    sim_sa = sim_sa[sim_sa['hours_per_month_by_activity'] != 0]

    norm = pd.concat([
        test_normality_log(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, log=False),
        test_normality_log(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, log=True)
    ], sort=False) \
        .rename(columns={'y': 'attribute'}) \
        .merge(season_order, on='season', how='left')

    """
    log_norm = test_normality_log(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, log=True)
    norm = test_normality_log(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, log=False)
    norm = s_merge(norm, log_norm, on=['grower_data', 'y', 'season', 'activity'], how='left', validate='1:1',
                   left_name='norm', right_name='log_norm').rename(columns={'y': 'attribute'})
    """
    norm[['season', 'activity', 'normal', 'log_normal']] = norm[['season', 'activity', 'normal', 'log_normal']].fillna('n/a')

    conds = [(norm['season']!='n/a') & (norm['activity']!='n/a'), norm['season']!='n/a']
    choices = ['by grower, season, activity', 'by grower, season']
    norm['specificity'] = np.select(conds, choices, default='by grower')
    norm = norm.sort_values(by=['grower_data', 'specificity', 'attribute', 'season_order', 'activity']).replace('0n/a', 'n/a')

    norm.to_csv(OUTPUT_PATH + 'normality_tests.csv', index=False)


def test_corr_wrapper(gd_s):

    # combine results for unemployed, retired, other
    df = gd_s.copy()
    df['employment_status'] = df['employment_status'].replace({4:3, 5:3, 6:3})

    corr = pd.concat([
        test_corr(df, x='farm_size_by_season_ha', y='hours_per_month_at_site'),
        test_corr(df, x='farm_size_by_season_ha', y='hours_per_month_all_activities'),
        test_corr(df, x='employment_status', y='hours_per_month_at_site', test='kendall_tau'),
        test_corr(df, x='employment_status', y='hours_per_month_all_activities', test='kendall_tau'),
        test_corr(df, x='hours_per_month_at_site', y='hours_per_month_all_activities')
    ])
    corr['specificity'] = 'by season'
    corr.to_csv(OUTPUT_PATH + 'correlation_tests.csv', index=False)

# Input ****************************************************************************************************************

# Grower data
gd = pd.read_csv(INPUT_PATH + 'grower_data/grower_data_by_grower.csv')
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

# **********************************************************************************************************************

test_kruskal_wallace_wrapper(gd_s, gd_sa)

test_normality_wrapper(gd, gd_s, emp_s, emp_sa, sim_s, sim_sa, season_order)

test_corr_wrapper(gd_s)
