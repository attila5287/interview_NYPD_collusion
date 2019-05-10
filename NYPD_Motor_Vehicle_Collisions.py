import warnings
warnings.simplefilter('ignore')
import pandas as pd
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from NYPD_functions import dataFrameNYC


def solution01of08():
    """Q-1: What is the total number of persons injured in the dataset (up to December 31, 2018?)"""
    pass
    master_df = dataFrameNYC()
    focus_df = master_df[['DATE', 'NUMBER OF PERSONS INJURED']]
    focus_df['dat3'] = [date[-4:] for date in focus_df['DATE']]
    focus_df.drop(columns='DATE', inplace=True)
    out_df = focus_df.groupby('dat3').sum()[:'2018']
    print(out_df)
    out_num = out_df.sum()
    print()
    print("--q1 answer for total number of persons injured:")
    print(out_num)
    print()
    return None


# solution01of08()
# --------------------------------
# total number of persons injured:
# NUMBER OF PERSONS INJURED    368,034


def solution02of08():
    """
    Q-2: What proportion of all collisions in 2016 occurred in Brooklyn? 
    Only consider entries with a non-null value for BOROUGH.
    """
    pass
    master_df = dataFrameNYC()
    focus_df = master_df[['DATE', 'BOROUGH', 'UNIQUE KEY']]
    focus_df['ye4r'] = [date[-4:] for date in focus_df['DATE']]
    focus_df.drop(columns='DATE', inplace=True)
    gr0uped_df = focus_df.groupby(['ye4r', 'BOROUGH']).count()
    gr0uped_df.reset_index(inplace=True)
    gr0uped2016_df = gr0uped_df.loc[gr0uped_df['ye4r'] == '2016']
    TOTAL_2016 = gr0uped2016_df['UNIQUE KEY'].sum()
    gr0uped2016_df['ratioBoro'] = [
        float(count / TOTAL_2016).__round__(4)
        for count in gr0uped2016_df['UNIQUE KEY']
    ]
    gr0uped2016_df['perc2total'] = [
        '{:,.2f} %'.format(ratio * 100)
        for ratio in gr0uped2016_df['ratioBoro']
    ]

    # ------------------------------
    _out = gr0uped2016_df
    print(gr0uped2016_df['ratioBoro'].sum())
    print(_out)
    final_df = gr0uped2016_df.set_index('BOROUGH').drop(
        columns=['UNIQUE KEY', 'ye4r', 'ratioBoro']).T
    print(final_df)
    print()
    print('-- q2 answer below: ')
    print('Brooklyn to total number of collusions for 2016 below:')
    print(final_df['BROOKLYN'][0])
    return None


# solution02of08()
# brooklyn 30.96 percent


def solution03of08():
    """
    Q-3: What proportion of collisions in 2016 resulted in injury or death of a cyclist?
    """
    pass
    master_df = dataFrameNYC()
    focus_df = master_df[[
        'DATE', 'UNIQUE KEY', 'NUMBER OF CYCLIST INJURED',
        'NUMBER OF CYCLIST KILLED'
    ]]
    focus_df['y3ar'] = [date[-4:] for date in focus_df['DATE']]
    focus2016_df = focus_df.loc[focus_df['y3ar'] == '2016']
    focus2016_df = focus2016_df.drop(columns=['DATE'])
    inj_num = focus2016_df.loc[
        focus2016_df['NUMBER OF CYCLIST INJURED'] != 0].count()[0]
    print('number of collusions at least a cyclist injured')
    print(inj_num)
    print()
    killed_num = focus2016_df.loc[
        focus2016_df['NUMBER OF CYCLIST KILLED'] != 0].count()[0]
    print('number of collusions at least a cyclist killed')
    print(killed_num)
    print()
    total_num = focus2016_df.count()[0]
    print('number of collusions for whole year of 2016')
    print(total_num)
    print()
    print('number of collusions at least a cyclist either injured or killed')
    either_num = inj_num + killed_num
    print(either_num)
    print()
    print('ratio of either inj or killed')
    ratio_either = either_num / total_num
    print(ratio_either.__round__(2))
    print()
    print('-- q3 answer for percentage of either inj or killed')
    perc_either = ratio_either.__round__(4) * 100
    formatted_answer = '{:,.2f} %'.format(perc_either)
    print(formatted_answer)
    # ratio_inj = float(injured_num/total_num)
    # ratio_kill = float(killed_num/total_num)
    # print()
    return None


# solution03of08()
# percentage of either inj or killed is 2.17 %


def solution04of08():
    """
    Q-4: For each borough, compute the number of accidents per capita involving alcohol in 2017. Report the highest rate among the 5 boroughs. Use populations as given by
    https://en.wikipedia.org/wiki/Demographics_of_New_York_City.
    """
    pass
    master_df = dataFrameNYC()
    focus_df = master_df[['DATE', 'BOROUGH', 'CONTRIBUTING FACTOR VEHICLE 1']]
    focus_df['ye4r'] = [date[-4:] for date in focus_df['DATE']]
    focus_df.drop(columns='DATE', inplace=True)
    focusOnDate_df = focus_df.loc[focus_df['ye4r'] == '2017']
    focusOnFactor_df = focusOnDate_df.loc[
        focusOnDate_df['CONTRIBUTING FACTOR VEHICLE 1'] ==
        'Alcohol Involvement']
    focusOnFactor_df.drop(columns='ye4r', inplace=True)
    print(focusOnFactor_df.head())
    print()
    onlyCount_df = focusOnFactor_df.groupby('BOROUGH').count().T
    print(onlyCount_df)
    # -----------------------
    fileName_demographicNYC = 'nyc_demograph_2017.csv'
    demographic_df = dataFrameNYC(fileName_demographicNYC)
    demographic_df.set_index('Jurisdiction', inplace=True)
    demographic_df = demographic_df.T
    print(demographic_df)
    print()
    boro_list = list(demographic_df.columns)
    alcoholPerCapita_list = [
        int(onlyCount_df[boro.upper()]) / int(demographic_df[boro])
        for boro in boro_list
    ]
    print(alcoholPerCapita_list)
    # ----------------------
    alcoholPerCapita_df = pd.DataFrame({
        'bor0ugh': boro_list,
        'rat3AlcPerCap': alcoholPerCapita_list
    })
    print(alcoholPerCapita_df)
    print()
    final_df = alcoholPerCapita_df.groupby('bor0ugh').sum().T
    print(final_df)

    def bringHighestRate(df=final_df):
        pass
        _df = df.copy()
        boro = ''
        highest = 0.0
        for boro in _df.columns:
            ratio_per_capita = _df[boro][0]
            if float(ratio_per_capita) > highest:
                highest = ratio_per_capita
                boro_highest = boro
            else:
                pass
        print(boro_highest + " is the highest " + '{:.5f}'.format(highest) +
              " per capita")

        return None

    print(
        '---q4 answer for highest boro for collusion rate of alcohol inv per capita'
    )
    bringHighestRate()
    return None


solution04of08()
# bor0ugh           Bronx  Brooklyn  Manhattan   Queens  Staten Island
# rat3AlcPerCap  0.000179  0.000213   0.000147  0.00021       0.000202
# Brooklyn is the highest 0.00021 per capita
