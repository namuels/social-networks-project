"""
    File name: dd_trait_merge.py
    Authors: Natalie Samuels & Jack Jester-Weinstein
    Date created: 8/23/2018
    Date last modified: 11/07/2018
    Python Version: 3.7
    Description: A data pre-processing script that cleans and merges
    daily diary, network, and trait data from the SSNL Social Networks
    project in preparation for analysis.

    Dorm Key:
    1 - Winter Trancos
    2 - Spring Rinconada
    3 - Fall Trancos
    4 - Fall Rinconada

"""


import sys
import pandas as pd
import numpy as np
import re


def melt_variable(dd_data, variable, column_regex):
    """
    :param pd.DataFrame dd_data: original daily diary data as a data frame
    :param str variable: variable name as a string
    :param str column_regex: regular expression string denoting variable (category, emotion rating, or dormmate number)
    :return pd.DataFrame: melted variable data frame with columns for event number and valence

    """

    # creates melted variable data frame
    columns_to_melt = [column_name for column_name in dd_data.columns if re.match(column_regex, column_name)]
    dd_variable_data = pd.melt(dd_data, id_vars=['ID', 'Dorm', 'Day'], value_vars=columns_to_melt)

    # reorders rows by ID, then Dorm, then Day
    dd_variable_data = dd_variable_data.sort_values(by=['ID', 'Dorm', 'Day'])
    dd_variable_data = dd_variable_data.reset_index(drop=True)

    # creates variable data frame with columns for event number and valence
    event_details = dd_variable_data['variable'].str.extract(column_regex)

    # adds event number and valance columns to melted variable data frame and cleans resulting data frame
    dd_variable_data = pd.concat([dd_variable_data, event_details], axis='columns')
    dd_variable_data = dd_variable_data.rename(columns={'value': variable})
    del dd_variable_data['variable']

    return dd_variable_data


def clean_dd(dd_data):
    """
    :param pd.DataFrame dd_data: original daily diary data as a data frame
    :return pd.DataFrame: cleaned daily diary data (long form, with columns for event number and valence)

    Reshape the DD data set into one event per row using the following steps:

        1a. Melt each variable into a separate data frame with one event per row
        1b. Pull out event number and valence from variable names by declaring a function to extract from the name and
            mapping it to the variable-name column
        2. Merge the resulting data frames back into one

    """

    # Step 1a + 1b

    dd_cat_data = melt_variable(dd_data, 'Category', r'(?P<Valence>Pos|Neg)Ev(?P<EventNum>\d)_Cat')

    dd_emo_data = melt_variable(dd_data, 'EmoRating', r'(?P<Valence>Pos|Neg)Ev(?P<EventNum>\d)_Emo')

    dd_drm1_data = melt_variable(dd_data, 'TldDrm1', r'Drm(?P<Valence>Ps|Ng)EvTld(?P<EventNum>[123])_1')
    dd_drm2_data = melt_variable(dd_data, 'TldDrm2', r'Drm(?P<Valence>Ps|Ng)EvTld(?P<EventNum>[123])_2')
    dd_drm3_data = melt_variable(dd_data, 'TldDrm3', r'Drm(?P<Valence>Ps|Ng)EvTld(?P<EventNum>[123])_3')
    dd_drm4_data = melt_variable(dd_data, 'TldDrm4', r'Drm(?P<Valence>Ps|Ng)EvTld(?P<EventNum>[123])_4')
    dd_drm5_data = melt_variable(dd_data, 'TldDrm5', r'Drm(?P<Valence>Ps|Ng)EvTld(?P<EventNum>[123])_5')

    # naming convention for valence is inconsistent between variables
    # this changes valence variables from short to long format
    for drm_data in [dd_drm1_data, dd_drm2_data, dd_drm3_data, dd_drm4_data, dd_drm5_data]:
        drm_data['Valence'] = drm_data['Valence'].str.replace('Ps', 'Pos')
        drm_data['Valence'] = drm_data['Valence'].str.replace('Ng', 'Neg')

    # Step 2

    dd_data_clean = pd.merge(dd_cat_data, dd_emo_data, on=['ID', 'Dorm', 'Day', 'Valence', 'EventNum'])

    for drm_data in [dd_drm1_data, dd_drm2_data, dd_drm3_data, dd_drm4_data, dd_drm5_data]:
        dd_data_clean = pd.merge(dd_data_clean, drm_data, on=['ID', 'Dorm', 'Day', 'Valence', 'EventNum'])

    return dd_data_clean


def sum_drm(dd_data_clean):
    """
    :param pd.DataFrame dd_data_clean: cleaned daily diary data (long form, with columns for event number and valence)
    :return pd.DataFrame: daily diary data with column for total dormmates told and binary "got help" column

    """
    columns_to_sum = [column_name for column_name in dd_data_clean.columns if re.match(r'TldDrm\d', column_name)]
    dd_data_clean['NumDrmTld'] = dd_data_clean[columns_to_sum].sum(axis=1)
    dd_data_clean['GotHelp'] = np.where(dd_data_clean['NumDrmTld'] == 0, 0, 1)

    return dd_data_clean


def clean_trait(trait_data):
    """
    :param pd.DataFrame trait_data: original daily diary data as a data frame
    :return pd.DataFrame: trait data at time point 0 with relevant regressors only

    """
    trait_data = trait_data.loc[trait_data['time'] == 0]
    trait_data = trait_data[['ID', 'Gender_bin', 'IRQ_TN', 'IRQ_EN', 'IRQ_TP', 'IRQ_EP', 'IRI_EC', 'IRI_PT', 'IRI_PD',
                             'PSA_avg', 'NTB_avg', 'MCSDS_avg', 'CESD_avg', 'STAI_MEAN', 'PSS_avg', 'Loneliness_avg',
                             'PANAS_NegAvg', 'PANAS_PosAvg', 'SHS_avg', 'SWLS_avg', 'BIS_avg', 'BAS_drive',
                             'NPI15_avg', 'BFI_a', 'BFI_e', 'BFI_n']]

    return trait_data


def main():
    args = sys.argv[1:]
    dd_data_filepath = args[0]
    trait_data_filepath = args[1]
    network_data_filepath = args[2]

    dd_data = pd.read_csv(dd_data_filepath)
    trait_data = pd.read_csv(trait_data_filepath)
    network_data = pd.read_csv(network_data_filepath)

    # prepare daily diary data for merge
    dd_data_clean = clean_dd(dd_data)
    dd_data_final = sum_drm(dd_data_clean)

    # prepare trait data for merge
    trait_data_final = clean_trait(trait_data)

    dd_trait_data = pd.merge(dd_data_final, trait_data_final, on=['ID'])
    dd_trait_network_data = pd.merge(dd_trait_data, network_data, on=['ID'])
    dd_trait_network_data.to_csv(r'/Users/nsamuels/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Social_Networks_Project/Data_Analysis/dd_trait_nsamuels.csv')


if __name__ == '__main__':
    main()
