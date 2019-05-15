import warnings
warnings.simplefilter('ignore')
import pandas as pd
import os
import datetime as dt
import numpy as np
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


# d solution04of08()
# bor0ugh           Bronx  Brooklyn  Manhattan   Queens  Staten Island
# rat3AlcPerCap  0.000179  0.000213   0.000147  0.00021       0.000202
# Brooklyn is the highest 0.00021 per capita


def solution05of08():
    """
    SOLUTION FOR FIFTH OF EIGHT
    """

    def dataFram3NYC(FILE_NAME='NYPD_Motor_Vehicle_Collisions.csv',
                     use_parent_dir=False):
        """
        READS ~1,5 MILLION ROW CSV FILE, RETURNS MASTER DATAFRAME
        use_parent_dir will use parent directory as base directory
        """
        if use_parent_dir == True:
            CURRENTWRK_DIR = os.getcwd()
            BASE_DIR = os.path.dirname(CURRENTWRK_DIR)
        elif use_parent_dir == False:
            BASE_DIR = os.getcwd()

    #     STATIC_FOLDER = os.path.dirname(BASE_DIR)
        STATIC_FOLDER = ''
        # FILE_NAME = 'NYPD_Motor_Vehicle_Collisions.csv'
        PATH_COMPILED = os.path.join(BASE_DIR, STATIC_FOLDER, FILE_NAME)
        print('\n> > > > dataFrameNYC() now reading csv with file path below')
        print(PATH_COMPILED)
        print()
        _df = pd.read_csv(
            PATH_COMPILED, encoding="iso-8859-1", low_memory=False)
        col_list = list(_df.columns)
        col_ind3x = np.arange(len(col_list))
        col_dict = dict(zip(col_ind3x, col_list))
        print(col_dict)
        return _df

    def focusDataframer(columns2keep_list=[0, 3, 24, 25, 26, 27, 28]):
        """
        MODIFIED DATA FRAMER, PRINTS OUT ALL COLUMN INDEX/NAMES, 
        REQUIRES A LIST OF REQUESTED COLUMN INDEXES AS INTEGERS
        RETURNS A DATAFRAME WITH ONLY USER- LABELS
        """
        raw_df = dataFram3NYC(use_parent_dir=True)
        col_list = list(raw_df.columns)
        col_ind3x = np.arange(len(col_list))
        col_dict = dict(zip(col_ind3x, col_list))
        col_final = []
        for index in columns2keep_list:
            add_this_column = col_dict[index]
            col_final.append(add_this_column)
        col_final = [str(index) for index in col_final]
        _df = raw_df[col_final]
        print(col_final)
        return _df

    def dateFilterMyDataFrame(focus_df=focusDataframer(),
                              bring_all_records_for='2016'):
        """
        REQUIRES A DATAFRAME WITH A TARGET FIELD NAMED 'DATE',
        FILTERS DATE BY FOUR DIGIT YEAR 'YYYY' STRING INPUT
        """
        pass
        print('> > > dateFilterMyDataFrame() filtering all records by year : '
              + bring_all_records_for)
        #     focus_df = focusDataframer()
        focus_df['dat3'] = [date[-4:] for date in focus_df['DATE']]
        filtered_by_date_df = focus_df.loc[focus_df['dat3'] ==
                                           bring_all_records_for]
        return filtered_by_date_df.drop(columns='DATE')

    def numberVehiclesInvAccidents(
            dateFilteredDataFrame=dateFilterMyDataFrame(
                bring_all_records_for='2016')):
        """
        REQUIRES A DATA FILTERED DATAFRAME, CALCULATES NUM OF VEHICLES
        FOR THE PRE-FILTERED DATE. RETURNS INTEGER AND PRINTS LOGS
        """
        pass
        focus2016_df = dateFilterMyDataFrame(bring_all_records_for='2016')
        vector_df = focus2016_df.isna()
        vehicle_columns_list = vector_df.columns[1:6]
        colTrueSum_list = vector_df[vehicle_columns_list].sum()
        accumulator = 0
        for sum in colTrueSum_list:
            accumulator += sum
        totalVehicle_num = len(vector_df) * 5 - accumulator
        print()
        print(
            '> > >findNumberAccidentsInYear() calculated number of vehicles involved in accidents: {:,}'
            .format(totalVehicle_num))
        return int(totalVehicle_num)

    numberVehiclesInvAccidents()

    # Q-5: Obtain the number of vehicles involved in each collision in 2016. Group the collisions by zip code and compute the sum of all vehicles involved in collisions in each zip code, then report the maximum of these values.

    def numberOfVehiclesInvZIPCode(
            focus2016_df=dateFilterMyDataFrame(bring_all_records_for='2016')):
        """
        REQUIRES A DATE FILTERED DATAFRAME, CALCULATES NUM VEHICLES INVOLVED
        RETURNS A SORTED DATAFRAME, PRINTS TOP TEN ZIP CODE AS A DICT
        """
        grouped_df = focus2016_df.groupby('ZIP CODE').count()
        grouped_df['numVehicleInv'] = 0
        for column in vehicle_columns_list:
            grouped_df['numVehicleInv'] = grouped_df[
                'numVehicleInv'] + grouped_df[column]
        final_df = grouped_df[['numVehicleInv']]
        finalSorted_df = final_df.sort_values(
            by='numVehicleInv', ascending=False)
        final_list = grouped_df['numVehicleInv']
        focus2016_df = dateFilterMyDataFrame(bring_all_records_for='2016')
        grouped_df = focus2016_df.groupby('ZIP CODE').count()
        grouped_df['numVehicleInv'] = 0
        for column in vehicle_columns_list:
            grouped_df['numVehicleInv'] = grouped_df[
                'numVehicleInv'] + grouped_df[column]
        final_df = grouped_df[['numVehicleInv']]
        finalSorted_df = final_df.sort_values(
            by='numVehicleInv', ascending=False)
        print(
            '> > >numberOfVehiclesInvZIPCode() calculated number of vehicle involved in acc per ZIP code  :'
        )
        promptAFewZip_dict = finalSorted_df.head().to_dict()
        print(promptAFewZip_dict)
        #     final_list = finalSorted_df['numVehicleInv']
        return finalSorted_df

    # numberOfVehiclesInvZIPCode()

    def mungleData4PlotTop5ZIPVehCnt(
            focus2016_df=dateFilterMyDataFrame(bring_all_records_for='2016')):
        """
        REQUIRES DATE FILTERED DATAFRAME, DATA MUNGLING FOR DATA VISUALIZATION
        """
        grouped_df = focus2016_df.groupby('ZIP CODE').count()
        grouped_df['numVehicleInv'] = 0
        vehicle_columns_list = grouped_df.columns
        for column in vehicle_columns_list:
            grouped_df['numVehicleInv'] = grouped_df[
                'numVehicleInv'] + grouped_df[column]
        final_df = grouped_df[['numVehicleInv']]
        finalSorted_df = final_df.sort_values(
            by='numVehicleInv', ascending=False)
        final_list = grouped_df['numVehicleInv']
        focus2016_df = dateFilterMyDataFrame(bring_all_records_for='2016')
        grouped_df = focus2016_df.groupby('ZIP CODE').count()
        grouped_df['numVehicleInv'] = 0
        for column in vehicle_columns_list:
            grouped_df['numVehicleInv'] = grouped_df[
                'numVehicleInv'] + grouped_df[column]
        final_df = grouped_df[['numVehicleInv']]
        finalSorted_df = final_df.sort_values(
            by='numVehicleInv', ascending=False)
        zip2Nhood_list = [
            'Kings', 'Long Island', 'BergenBeach', 'Queens', 'Kings_'
        ]
        topFive_df = finalSorted_df.head()
        topFive_df['neighb0urhood'] = [
            str(hoodname) for hoodname in zip2Nhood_list
        ]
        dataPlotFive_df = topFive_df.reset_index().drop(
            columns='ZIP CODE').set_index('neighb0urhood')
        print()
        print('> > > mungleData4PlotTop5ZIPVehCnt() adds names to ZIP codes')
        print(dataPlotFive_df.head().to_dict())
        return dataPlotFive_df

    mungleData4PlotTop5ZIPVehCnt()

    def barChartTop5ZIPVehCnt(dataPlotFive_df=mungleData4PlotTop5ZIPVehCnt()):
        """
        REQURES DATA FRAME BY mungleData4PlotTop5ZIPVehCnt() AND SHOWS A BAR CHART
        """
        x_bar = list(dataPlotFive_df.index)
        y_bar = dataPlotFive_df['numVehicleInv']
        plt.barh(
            x_bar,
            y_bar,
            color='coral',
            linewidth=2,
            edgecolor='lightgrey',
            alpha=0.80)
        plt.savefig('Q5_ZIPCode_numberVehicles.png')
        plt.show()
        plt.savefig('Q5_ZIPCode_numberVehicles.png')
        print()
        print(
            'barChartTop5ZIPVehCnt() saved figure under file named Q5_ZIPCode_numberVehicles.png'
        )
        return None

    barChartTop5ZIPVehCnt(dataPlotFive_df=mungleData4PlotTop5ZIPVehCnt())
    return None


solution05of08()