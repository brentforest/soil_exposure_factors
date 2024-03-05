import numpy as np
import pandas as pd
import itertools
from prep_grower_data import *
from utilities import *

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

# Note: I had previously converted soil ingestion rates to kg, but scientific notation makes it harder to interpret results tables.
# Instead I just divide the avg daily dose by 1,000,000.
CONCENTRATION = 400 # mg/kg soil
SI_DAILY = 378 # mg/day
SI_OUTDOOR = 45.25 # mg/hour
SI_INDOOR = 1.375 # mg/hour

WEEKS_PER_MONTH = 4.35
DAYS_PER_MONTH = 30.45
HOURS_PER_MONTH = DAYS_PER_MONTH * 24

N_SIMULATIONS = 5000
RANDOM_SEED = 5
SEASON_ORDER = ['Spring', 'Summer', 'Fall', 'Winter']

# File paths
OUTPUT_PATH = '../output/data/exposure/'

def compute_body_weight_nhanes(nh_bw, nh_age):
# Merge and filter nhanes body weight and age, for use in monte carlo

    nh_bw = nh_bw[nh_bw['BMXWT'].notna()]
    nh_age = nh_age[nh_age['RIDAGEYR'].notna()]
    nh_age = nh_age[nh_age['RIDAGEYR'] >= 21] # Consistent with how EFH defines adult for dermal
    nh_bw = nh_bw.merge(nh_age, on='SEQN', how='inner', validate='1:1')

    # Weights must sum to 1.0 for use in np.random.choice
    nh_bw['weight_total'] = nh_bw['WTMECPRP'].sum()
    nh_bw['weight'] = nh_bw['WTMECPRP'] / nh_bw['weight_total']

    return nh_bw

def compute_avg_daily_dose(method, gro, sea, act):
# Compute avg daily dose in mg/kg BW day

    # Merge grower data with seasonal data
    df = s_merge(gro, sea, on='grower_id', how='left', validate='1:m')

    # Add global vars to dataframe so they show up in exported file
    df['concentration_mg/kg'] = float(CONCENTRATION)
    df['si_daily_mg/day'] = float(SI_DAILY)
    df['si_outdoor_mg/hour'] = float(SI_OUTDOOR)
    df['si_indoor_mg/hour'] = float(SI_INDOOR)

    if method == 'method_1':

        df = df[['grower_id', 'season', 'concentration_mg/kg', 'body_weight_kg',
                      'days_per_month_at_site', 'si_daily_mg/day']]

        df['ef_method_1'] = df['days_per_month_at_site'] / DAYS_PER_MONTH  # unitless
        df['avg_daily_dose_method_1'] = df['concentration_mg/kg'] * df['si_daily_mg/day'] * df['ef_method_1'] / df['body_weight_kg'] / 1000000
        return df

    elif method == 'method_2':

        df = df[['grower_id', 'season', 'concentration_mg/kg', 'body_weight_kg',
             'hours_per_month_at_site', 'si_outdoor_mg/hour', 'si_indoor_mg/hour']]

        df['ef_method_2_outdoor'] = df['hours_per_month_at_site'] / HOURS_PER_MONTH # unitless
        df['avg_daily_dose_method_2_outdoor'] = df['concentration_mg/kg'] * df['si_outdoor_mg/hour'] * df['ef_method_2_outdoor'] / df['body_weight_kg'] / 1000000 * 24 # x24 converts avg hourly dose to avg daily dose

        df['ef_method_2_indoor'] = 1 - df['ef_method_2_outdoor']
        df['avg_daily_dose_method_2_indoor'] = df['concentration_mg/kg'] * df['si_indoor_mg/hour'] * df['ef_method_2_indoor'] / df['body_weight_kg'] / 1000000 * 24

        df['avg_daily_dose_method_2'] = df['avg_daily_dose_method_2_outdoor'] + df['avg_daily_dose_method_2_indoor']
        return df

    elif method=='method_3':

        # Merge w/activity data
        df = s_merge (df, act, on=['grower_id', 'season'], how='left', validate='1:m')

        df = df[['grower_id', 'season', 'activity', 'concentration_mg/kg', 'body_weight_kg', 'hours_per_month_by_activity',
             'si_by_activity_min_mg/hour', 'si_by_activity_max_mg/hour','si_by_activity_mg/hour', 'si_indoor_mg/hour']]

        # To prevent double-counting, we can't calculate indoor exposure until AFTER grouping by season
        df['ef_method_3_outdoor'] = df['hours_per_month_by_activity'] / HOURS_PER_MONTH  # unitless
        df['avg_daily_dose_method_3_outdoor'] = df['concentration_mg/kg'] * df['si_by_activity_mg/hour'] * df['ef_method_3_outdoor'] / df['body_weight_kg'] / 1000000 * 24 # x24 converts avg hourly dose to avg daily dose

        # Group method 3 by season
        df_by_sea = df.groupby(['grower_id', 'body_weight_kg', 'season', 'concentration_mg/kg', 'si_indoor_mg/hour'])[
            ['hours_per_month_by_activity', 'ef_method_3_outdoor', 'avg_daily_dose_method_3_outdoor']].sum().reset_index() \
            .rename(columns={'hours_per_month_by_activity': 'hours_per_month_all_activities'})

        df_by_sea['ef_method_3_indoor'] = (1 - df_by_sea['ef_method_3_outdoor'])
        df_by_sea['avg_daily_dose_method_3_indoor'] = df_by_sea['concentration_mg/kg'] * df_by_sea['si_indoor_mg/hour'] * df_by_sea['ef_method_3_indoor'] / df_by_sea['body_weight_kg'] / 1000000 * 24

        df_by_sea['avg_daily_dose_method_3'] = df_by_sea['avg_daily_dose_method_3_outdoor'] + df_by_sea['avg_daily_dose_method_3_indoor']
        return(df, df_by_sea)

def generate_simulated_growers(nh_bw, sea, act, si_by_activity):
# Monte Carlo simulation
# https://pbpython.com/monte-carlo.html

    # Set random seed
    # Note you MUST use "np.random," not just "ranodom.seed"! Without the "np." it doesn't work and you will get different results every time!
    print('Random seed:', RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print('Running simulations, N=', N_SIMULATIONS)

    # Create empty dataframes w/permutations of growers, seasons, activities
    # https://stackoverflow.com/questions/45672342/create-a-dataframe-of-permutations-in-pandas-from-list
    grower_ids = list(range(1, N_SIMULATIONS + 1))

    activities = act['activity'].drop_duplicates().to_list()
    mc_gro = pd.DataFrame(grower_ids, columns=['grower_id'])
    mc_sea = pd.DataFrame(list(itertools.product(*[grower_ids, SEASON_ORDER])), columns=['grower_id', 'season'])
    mc_act = pd.DataFrame(list(itertools.product(*[grower_ids, SEASON_ORDER, activities])),
                          columns=['grower_id', 'season', 'activity'])

    # Get si vars
    mc_act = s_merge(mc_act, si_by_activity, on='activity', how='left', validate='m:1')

    # Sample by grower (see notes docs for details on NHANES bodyweight sampling)
    mc_gro['body_weight_kg'] = np.random.choice(nh_bw['BMXWT'], size=N_SIMULATIONS, p=nh_bw['weight'])

    for s in SEASON_ORDER:

        sea_a = s_filter(sea, col='season', list=[s])
        conds = (mc_sea['season'] == s)
        mc_sea.loc[conds, 'days_per_month_at_site'] = np.random.choice(sea_a['days_per_month_at_site'], size=N_SIMULATIONS)
        mc_sea.loc[conds, 'hours_per_month_at_site'] = np.random.choice(sea_a['hours_per_month_at_site'], size=N_SIMULATIONS)

        for a in activities:

            # Sample by activity
            act_a = act[(act['activity'] == a) & (act['season'] == s)]

            # note we are KEEPING data where the frequency of an activity = 0.
            # Same as with empirical results, some growers don't perform an activity during a given season.
            # Non-doers get filtered out before plotting histograms.
            conds = (mc_act['activity'] == a) & (mc_act['season'] == s)
            mc_act.loc[conds, 'hours_per_month_by_activity'] = np.random.choice(act_a['hours_per_month_by_activity'], size=N_SIMULATIONS)

            # For method 3 only: Generate random soil ingestion value
            min = act_a['si_by_activity_min_mg/hour'].iat[0]
            max = act_a['si_by_activity_max_mg/hour'].iat[0]
            mc_act.loc[conds, 'min'] = min
            mc_act.loc[conds, 'max'] = max
            mc_act.loc[conds, 'si_by_activity_mg/hour'] = np.random.uniform(min, max, size=N_SIMULATIONS)

    # Convert zipped tuple into individual columns
    #mc[['days_per_month', 'hours_per_month_by_activity']] = mc['freq_vars'].apply(pd.Series)

    return (mc_gro, mc_sea, mc_act)


def compute_annual_exposure(df, results_cols, groupby, pivot_col, season_count):

    # Multiply by number of days in each season
    # https://www.npr.org/templates/story/story.php?storyId=5335287
    for col in results_cols:
        df.loc[df['season'] == 'Spring', col] *= 93
        df.loc[df['season'] == 'Summer', col] *= 94
        df.loc[df['season'] == 'Fall', col] *= 90
        df.loc[df['season'] == 'Winter', col] *= 89

    # Only keep growers who completed questionnaires in all four seasons, per SL request 2023-08-23
    df['season_count'] = df.groupby('grower_id')['season'].transform('count')
    df = df[df['season_count']==season_count]

    df = df.fillna(0).groupby(groupby)[results_cols].sum().reset_index()
    print('Length of annual dataframe:',len(df))
    return(df)


def compute_output_exposure(gro, sea, act, emp_sim):

    # Compute exposure
    m1_by_sea = compute_avg_daily_dose('method_1', gro, sea, act)
    m2_by_sea = compute_avg_daily_dose('method_2', gro, sea, act)
    (m3_by_sea_act, m3_by_sea) = compute_avg_daily_dose('method_3', gro, sea, act)

    # Output method 3 outdoor only, by activity
    m3_by_sea_act.to_csv(OUTPUT_PATH + emp_sim + '_exposure_by_season_activity_method_3_outdoor_only.csv', index=False)

    # Merge to allow for comparisons across methods
    # TODO: caution: some columns might exist in more than one dataset; only one of each gets included in the final merged dataframe
    all_by_sea = s_merge(m1_by_sea, m2_by_sea, on=['grower_id', 'body_weight_kg', 'season'], how='left', validate='1:1') \
        .pipe(s_merge, m3_by_sea, on=['grower_id', 'season'], how='outer', validate='1:1')

    all_by_sea.to_csv(OUTPUT_PATH + emp_sim + '_exposure_by_season.csv', index=False)

    # Annual totals, for stacked bar charts
    if emp_sim == 'empirical':
        print('\nComputing annual exposure:')
        results_cols = ['avg_daily_dose_method_1', 'avg_daily_dose_method_2', 'avg_daily_dose_method_3',
                        'avg_daily_dose_method_2_indoor', 'avg_daily_dose_method_2_outdoor',
                        'avg_daily_dose_method_3_indoor', 'avg_daily_dose_method_3_outdoor',
                        'hours_per_month_all_activities', 'hours_per_month_at_site']
        compute_annual_exposure(all_by_sea, results_cols=results_cols,
                            groupby=['grower_id', 'season'], pivot_col='season', season_count=4) \
            .to_csv(OUTPUT_PATH + emp_sim + '_exposure_annual_by_season.csv', index=False)

        compute_annual_exposure(m3_by_sea_act, results_cols=['avg_daily_dose_method_3_outdoor'],
                            groupby=['grower_id', 'activity'], pivot_col='activity', season_count=24) \
            .to_csv(OUTPUT_PATH + emp_sim + '_exposure_annual_by_activity.csv', index=False)


# Input ****************************************************************************************************************

# Body weight and demographic data from NHANES, monte carlo only; 2017-2020 data:
# https://wwwn.cdc.gov/Nchs/Nhanes/Search/DataPage.aspx?Component=Examination&Cycle=2017-2020
nh_bw = pd.read_sas('../input/nhanes/P_BMX.XPT')[['SEQN', 'BMXWT']]
nh_age = pd.read_sas('../input/nhanes/P_DEMO.XPT')[['SEQN', 'RIDAGEYR', 'RIDAGEMN', 'WTINTPRP', 'WTMECPRP']]

# Soil ingestion values
si_by_activity = pd.read_excel('../input/soil_ingest_by_activity.xlsx', sheet_name='si', skiprows=2)[
    ['activity', 'si_by_activity_mg/hour', 'si_by_activity_min_mg/hour', 'si_by_activity_max_mg/hour']] # for empirical method 3

# Test data to verify that weighted sampling works in the monte carlo
nh_bw_test = pd.read_csv('../input/nhanes/body_weight_test.csv')

# **********************************************************************************************************************

# Prepare grower data; this function is imported a separate script because it is also used by other scripts
(gro, sea, act) = prep_grower_data()
act = s_merge(act, si_by_activity, on='activity', how='left', validate='m:1')

# Estimate exposure, output, plot
compute_output_exposure(gro, sea, act, emp_sim='empirical')

# Prepare NHANES bodyweight data for Monte Carlo; plot
nh_bw = compute_body_weight_nhanes(nh_bw, nh_age)
nh_bw.to_csv(OUTPUT_PATH + 'body_weight_nhanes.csv', index=False)

# Generate simulated grower data
(mc_gro, mc_sea, mc_act) = generate_simulated_growers(nh_bw, sea, act, si_by_activity)

# Estimate exposure, output
compute_output_exposure(mc_gro, mc_sea, mc_act, emp_sim='simulated')