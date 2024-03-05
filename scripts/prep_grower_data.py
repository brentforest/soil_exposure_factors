import numpy as np
import pandas as pd
from utilities import *
import math

pd.options.display.max_columns = 999
pd.options.mode.chained_assignment = None

PATH = '../output/data/grower_data/'
WEEKS_PER_MONTH = 4.35

def prep_by_season_activity(df, recoded_activities, recoded_attributes):
# Prep vars specific to grower, season, and activity, e.g., time spent on activities, attire worn
# Converts from wide format to less-wide; index = grower, season, activity

    # Convert wide to long
    df = df.melt(id_vars=['grower_id', 'season'], var_name='attribute')

    # Create a separate var for activity names
    activities = ['bed_prep', 'harvest', 'plant_seed', 'plant_transplant', 'water', 'weed']
    conds = []
    for a in activities:
        conds.append(df['attribute'].str.contains(a))
    df['activity'] = np.select(conds, activities, default='other')

    # Drop variables that don't apply to a specific activity
    df = df[df['activity'] != 'other']

    # Drop activity names from variables (since activity is now a separate var)
    for a in activities:
        df['attribute'] = df['attribute'].str.replace(a + '_', '')

    # Merge w/new variable names
    df = s_merge(df, recoded_attributes, on='attribute', how='right')\
           .pipe(s_merge, recoded_activities, on='activity', how='left')

    # Drop attire vars
    #df = s_filter(df, col='attribute_recoded', excl_str='attire')

    # Pivot attribute
    df = s_pivot(df, idx=['grower_id', 'season', 'activity_recoded'], cols=['attribute_recoded'], vals=['value'])
    df = df.rename(columns={'activity_recoded': 'activity'})

    # TODO: check w/Sara about this
    # Drop rows with days/month data but no hours/day data, or vice versa (we need both to compute hours/month)
    # This should only drop 1 row: grower 1, winter, bed prep; leaving 743 rows instead of 744
    df = df[((df['days_per_month_by_activity'].notna()) & (df['hours_per_day_by_activity'].notna())) |
        ((df['days_per_month_by_activity'].isna()) & (df['hours_per_day_by_activity'].isna()))]

    # NAN values for activity data = grower did not engage in an activity during that season
    # Missing rows = grower did not farm during that season
    # See "notes_missing_data" for more details
    df['hours_per_day_by_activity'] = df['hours_per_day_by_activity'].fillna(0)
    df['days_per_month_by_activity'] = df['days_per_month_by_activity'].fillna(0)

    # Compute hours per month
    df['hours_per_month_by_activity'] = df['hours_per_day_by_activity'] * df['days_per_month_by_activity']

    # Compute hours per month in contact w/soil
    # (We're not currently using this for exposure factors)
    #df['hours_per_month_contact'] = df['hours_per_month'] * (df['contact_%'] / 100)

    df['specificity'] = 'by grower, season, activity'

    return df


def prep_by_season(df, si_codes, act):
# Prep vars that are specific to grower and season, e.g., soil ingestion

    # Filter, drop duplicate values
    df = df[['grower_id', 'season', 'farm_size', 'employment_status', 'soil_ingest', 'soil_face_days_per_month', 'soil_ingest_days_per_month', 'soil_ingest_amt_code',
             'produce_consume_freq', 'consume_snacks_freq', 'hours_per_day', 'days_per_week']].drop_duplicates()

    print('Prepping grower data by season, N =',len(df))

    # Convert farm size acres to hectares. Note this may vary by season, e.g., if a grower switches farms
    df['farm_size_by_season_ha'] = df['farm_size'] * 0.404686

    # Compute hours at the site - not specific to any activity
    # "For each day you are on site this season, how many hours per day are you typically present at your farm/garden?"
    df = df.rename(columns={'hours_per_day': 'hours_per_day_at_site', 'days_per_week': 'days_per_week_at_site'})
    df['days_per_month_at_site'] = df['days_per_week_at_site'] * WEEKS_PER_MONTH
    df['hours_per_month_at_site'] = df['hours_per_day_at_site'] * df['days_per_month_at_site']

    # Merge with soil ingest amount conversion from numbers to amounts
    df = df.merge(si_codes, on='soil_ingest_amt_code', how='left', validate='m:1')

    # Add sum of hours_per_month_by_activity to seasonal data, compute difference
    hours_sea = act.groupby(['grower_id', 'season'])['hours_per_month_by_activity'].sum().rename('hours_per_month_all_activities').reset_index()
    df = s_merge(df, hours_sea, how='left', on=['grower_id', 'season'])
    df['hours_per_month_diff'] = df['hours_per_month_at_site'] - df['hours_per_month_all_activities']

    df['specificity'] = 'by grower, season'

    return df


def prep_by_grower(df, bw, age):
# Prep vars that are specific to grower, e.g., age, sex, bodyweight

    # Farm size is reported by seasons, and some farmers report slightly different farm sizes by season, so use the avg:
    df['farm_size_mean_acres'] = df.groupby(['grower_id'])['farm_size'].transform('mean')
    df['farm_size_mean_ha'] = df['farm_size_mean_acres'] * 0.404686

    # Filter, drop duplicate values
    df = df[['grower_id', 'sex', 'year_of_birth', 'farm_size_mean_ha']].drop_duplicates()

    print('Prepping grower data by grower, N =', len(df))

    # One respondent said they are a 'boomer' and didn't provide DOB; use midpoint for baby boomers (1946-1964)
    df.loc[df['year_of_birth'] == '.', 'year_of_birth'] = '1955'

    # Compute age
    df['age'] = 2020 - df['year_of_birth'].astype(int)

    # Compute bodyweight based on age
    df = s_merge(df, age, on='age', how='left', validate='m:1') \
        .pipe(s_merge, bw, on=['sex', 'age_lower_bracket'], how='left')

    df['specificity'] = 'by grower'

    return df

def prep_grower_data():

    # Input ************************************************************************************************************

    raw = pd.read_excel('../input/questionnaire_2021_08_16.xlsx', sheet_name='data')

    # Recoded var names
    recoded_activities = pd.read_csv('../input/recoded_activities.csv')
    recoded_attributes = pd.read_csv('../input/recoded_attributes.csv')

    # Recoded soil ingestion amounts
    si_codes = pd.read_csv('../input/soil_ingest_amount_codes.csv')

    # Body weight data and age brackets from exposure factors handbook
    bw = pd.read_csv('../input/body_weights.csv', skiprows=1)[['sex', 'age_lower_bracket', 'body_weight_kg']]
    age = pd.read_csv('../input/age_brackets.csv')

    season_order = pd.DataFrame([['All seasons', 0], ['Spring', 1], ['Summer', 2], ['Fall', 3], ['Winter', 4]],
                                columns=['season', 'season_order'])

    # ******************************************************************************************************************

    # Recode season names
    raw['season'] = raw['season'].str.replace('_', ' ').str.capitalize()

    # Prep data by grower, season, activity.
    act = prep_by_season_activity(raw, recoded_activities, recoded_attributes)
    act.to_csv(PATH + 'grower_data_by_season_activity.csv', index=False)

    # Prep data by grower and season.
    sea = prep_by_season(raw, si_codes, act)
    sea.to_csv(PATH + 'grower_data_by_season.csv', index=False)

    # Prep data by grower.
    gro = prep_by_grower(raw, bw, age)
    gro.to_csv(PATH + 'grower_data_by_grower.csv', index=False)

    # Merge and output a version for supplementary data only (unused)
    """
    s_merge(gro, sea, on='grower_id', how='left', validate='1:m') \
        .pipe(s_merge, act, on=['grower_id', 'season'], how='left', validate='1:m') \
        .pipe(s_merge, season_order, on='season', how='left') \
        .sort_values(by=['grower_id', 'season_order', 'activity']).drop(columns='season_order') \
        .to_csv(PATH + 'grower_data_all.csv', index=False)
    """

    # Long form version
    gro_long = gro.copy()
    gro_long[['season', 'activity']] = ['n/a', 'n/a']
    sea_long = sea.copy()
    sea_long['activity'] = 'n/a'
    id_vars = ['specificity', 'grower_id', 'season', 'activity']
    pd.concat([gro_long.melt(id_vars=id_vars, var_name='attribute'),
                          sea_long.melt(id_vars=id_vars, var_name='attribute'),
                          act.melt(id_vars=id_vars, var_name='attribute')
              ]) \
        .merge(season_order, on='season', how='left') \
        .to_csv(PATH + 'grower_data_all_long.csv', index=False)

    return (gro, sea, act)