import warnings
warnings.simplefilter('ignore')
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn import (
    datasets,
    linear_model
)
from sklearn.metrics import (
    mean_squared_error, 
    r2_score
)
from NYPD_functions import (
    dataFram3NYC, 
    dataFram3NYC,
    focusDataframer,
    dateFilterMyDataFrame
)
from generate_color4_plot import (
    generateRandomDark_c0lor,
    generateRandomLight_c0lor,
    generateRandomMatplotlib_c0lor,
    cut_st4r
    )


def solution01of08():
    """Q-1: What is the total number of persons injured in the dataset (up to December 31, 2018?)"""
    pass
    master_df = dataFram3NYC()
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
    master_df = dataFram3NYC()
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
    master_df = dataFram3NYC()
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
    master_df = dataFram3NYC()
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
    demographic_df = dataFram3NYC(fileName_demographicNYC)
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
# solution04of08()
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
        print('\n> > > > dataFram3NYC() now reading csv with file path below')
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
        
        print()
        print(
            'barChartTop5ZIPVehCnt() saved figure under file named Q5_ZIPCode_numberVehicles.png'
        )
        return None

    barChartTop5ZIPVehCnt(dataPlotFive_df=mungleData4PlotTop5ZIPVehCnt())
    return None
# solution05of08()

def solution06of08():

    def gr0upYearByCollusionCount(focus_df = focusDataframer(columns2keep_list = [0,23])):
        """
        REQUIRES A DATAFRAME WITH A TARGET FIELD NAMED 'DATE',
        GROUP YEAR BY COLLUSION COUNT
        """
        pass
        print()
        print('> > > > gr0upYearByCollusionCount() calculated collusion count under year: ')
        focus_df['ye4r'] = [date[-4:] for date in focus_df['DATE']]
        gr0uped_df = focus_df.groupby('ye4r').count().drop(columns = 'DATE')
        gr0uped_df['c0llusionCount'] = [
            collusionCount
            for collusionCount in gr0uped_df['UNIQUE KEY']
        ]
        gr0uped_df = gr0uped_df.drop(columns='UNIQUE KEY')
        check_row_total = gr0uped_df.sum()[0]
        print('> > > > check if sum of all yearly collusion counts add up total row number {:,}'.format(check_row_total))
        print('> > > >                                                                     1,487,383 expected [?]')
        return gr0uped_df

    def yearLabeledCollusionCountDF(gr0uped_df=gr0upYearByCollusionCount(),
                                    year_first=2013,
                                    year_last=2018):
        """
        REQUIRES GROUPED DATAFRAME WITH YEAR ON INDEX AND COLLUSION COUNT AS VALUES
        """
        pass
        print()
        print(
            '> > > yearLabeledCollusionCountDF() returns a _df with below years on label: '
        )
        year_last += 1
        yrs_list = [
            int(year) for year in np.arange(int(year_first), int(year_last))
        ]
        print(yrs_list)
        yearLabel_df = gr0uped_df.T
        yearStr_list = [str(year) for year in yrs_list]
        return yearLabel_df[yearStr_list].T

    def describeRegressionLine(gr0up_df = yearLabeledCollusionCountDF(),
                            year_first = 2013,
                            year_last = 2018):
        """
        Q-6: Consider the total number of collisions each year from 2013-2018. 
        * Is there an apparent trend? 
        * Fit a linear regression for the number of collisions per year and report its slope.
        """
        # Create linear regression object
        regr = linear_model.LinearRegression(fit_intercept=True)
        # ------------------------------------
        int_for_x_axis = [int(year)for year in gr0up_df.index]
        y_axis = pd.Series(gr0up_df['c0llusionCount'])
        year_last += 1
        x_ax1s = np.arange(int(year_first), int(year_last))
        x_ax2s = x_ax1s.reshape(-1, 1)
        #  x_ax2s should be reshaped for R sq matrix
        regr.fit(x_ax2s, y_axis)
        y_predict = regr.predict(x_ax2s)
        # predicted number can not have decimals thus to be rounded to an integer
        y_pred1ct = [int(predicted.__round__(0)) for predicted in y_predict]
        print('> > > > describeRegressionLine() provides brief info on plotted regression line')
        print('> > > > Coefficients: ', regr.coef_[0])
        print('> > > > Intercept: ', regr.intercept_)
        label_coef = '{:,.0f}'.format(regr.coef_[0])
        label_intercept = '{:,.0f}'.format(regr.intercept_)
        print('> > > > * Equation of regression line ', label_coef, 'x + ' , label_intercept)
        print("> > > > Mean squared error: %.2f" % mean_squared_error(y_axis, y_predict))
        print('> > > > R-2-Variance score: %.2f' % r2_score(y_axis, y_predict))
        print('> > > >   ')
        print('> > > > *: positive trend observed meaning that every year')
        print('> > > >    there is an increase in number of collusions aprx 6,500')
        print()
        return None


    def describeAnotherRegressionLine(gr0up_df = yearLabeledCollusionCountDF(),
                            year_first = 2013,
                            year_last = 2018):
        """
        MODIFIED  VERSION FOR FORCED INTERCEPT ZERO, TO CHECK/SHOW TREND IS POSITIVE 
        Q-6: Consider the total number of collisions each year from 2013-2018. 
        * Is there an apparent trend? 
        * Fit a linear regression for the number of collisions per year and report its slope.
        """
        # Create linear regression object
        regr = linear_model.LinearRegression(fit_intercept=False)
        # ------------------------------------
        int_for_x_axis = [int(year)for year in gr0up_df.index]
        y_axis = pd.Series(gr0up_df['c0llusionCount'])
        year_last += 1
        x_ax1s = np.arange(int(year_first), int(year_last))
        x_ax2s = x_ax1s.reshape(-1, 1)
        #  x_ax2s should be reshaped for R sq matrix
        regr.fit(x_ax2s, y_axis)
        y_predict = regr.predict(x_ax2s)
        # predicted number can not have decimals thus to be rounded to an integer
        y_pred1ct = [int(predicted.__round__(0)) for predicted in y_predict]
        print('> > > > describeRegressionLine() provides brief info on plotted regression line')
        print('> > > > Coefficients: ', regr.coef_[0])
        print('> > > > Intercept: ', regr.intercept_)
        label_coef = '{:,.0f}'.format(regr.coef_[0])
        label_intercept = '{:,.0f}'.format(regr.intercept_)
        print('> > > > * Equation of regression line ', label_coef, 'x + ' , label_intercept)
        print("> > > > Mean squared error: %.2f" % mean_squared_error(y_axis, y_predict))
        print('> > > > R-2-Variance score: %.2f' % r2_score(y_axis, y_predict))
        print('> > > > *: positive trend means prediction oversees increase every year')
        print('> > > > **  very low r-2-score thus not useful')
        print()
        return None



    def prompt_N_plot_regressionLine(
        gr0up_df=yearLabeledCollusionCountDF(),
        year_first=2013,
        year_last=2018,
        c0lor=generateRandomDark_c0lor(),
        c0lor2=generateRandomLight_c0lor(),
        c0lor3=generateRandomDark_c0lor(),
        c0lor4=generateRandomDark_c0lor()
    ):
        """
        PLOTS REGRESSION LINES WITH EQUATIONS ON LABEL, SCATTERS ACTUAL DATA POINTS
        """
        pass
        describeRegressionLine()
        describeAnotherRegressionLine()
        y_axis = [int(collusionCount) for collusionCount in gr0up_df['c0llusionCount']]
        year_last += 1
        x_ax1s = np.arange(int(year_first), int(year_last))
        x_ax2s = x_ax1s.reshape(-1, 1)

        ###### FIRST SCATTER ACTUAL DATA POINTS #####
        plt.figure(figsize=(19, 10))
        plt.scatter(
            x_ax2s,
            y_axis,
            s=1200,
            color=c0lor,
            marker=cut_st4r(5),
            edgecolors=c0lor, 
            linewidth=3,
            alpha=0.5,
            label='Target: Number of Collusions'
        )
        ###### SECOND, FIT_INTERCEPT=FALSE ######
        # now we generate labels for regression lines plotted
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(x_ax2s, y_axis)
        y_predict = regr.predict(x_ax2s)
        # predicted number can not have decimals thus to be rounded to an integer
        y_pred1ct = [int(predicted.__round__(0)) for predicted in y_predict]
        label_coef = '{:,.0f}'.format(regr.coef_[0])
        label_intercept = '{:,.0f}'.format(regr.intercept_)
        lab3l = 'Predicted: RegLine ' + label_coef + '*X + ' + label_intercept
        plt.plot(
            x_ax2s,
            y_predict,
            color=c0lor3,
            marker='s',
            markersize= 25,
            markeredgewidth=5,
            markeredgecolor=c0lor3,
            markerfacecolor=c0lor2,
            linestyle='dashdot',
            linewidth=1,
            alpha=0.7,
            label=lab3l)
    #     ---------------------
        ###### THIRD, FITTED REGRESSION LINE #####
        r3gr = linear_model.LinearRegression(fit_intercept=True)
        r3gr.fit(x_ax2s, y_axis)
        y_pr3dict = r3gr.predict(x_ax2s)
        # predicted number can not have decimals thus to be rounded to an integer
        y_pr3d1ct = [int(predicted.__round__(0)) for predicted in y_pr3dict]
        label_co3f = '{:,.0f}'.format(r3gr.coef_[0])
        label_interc3pt = '{:,.0f}'.format(r3gr.intercept_)
        l4b3l = 'Predicted-Fitted: RegLine ' + label_co3f + '*X  ' + label_interc3pt
        plt.plot(x_ax2s,
                y_pr3dict,
                color=c0lor4,
                marker='o',
                markersize=30,
                markeredgewidth=7,
                markeredgecolor=c0lor2,
                markerfacecolor=c0lor4,
                linestyle='dotted',
                linewidth=1.5,
                alpha=0.9,
                label=l4b3l)
        plt.legend(loc='best')
        plt.rcParams["font.size"] = 14
        plt.title('Q6 REGRESSION LINE: Y=NUM-OF-COLLUSION, X=YEAR-2013 ')
        plt.xlabel('YEAR')
        plt.ylabel('NUMBER OF COLLUSION: PREDICTED VS TARGET')
        file2save = 'Q6_REGRESSION_LINES_NUM-COLLUSION_YEAR.png'
        plt.savefig(file2save)
        plt.show()
        return None

    prompt_N_plot_regressionLine()    
    
    return None
# solution06of08()


def solution07aof08():
    """
    Q-7: Do winter driving conditions lead to more multi-car collisions?
    * Compute the rate of multi car collisions as the proportion of the number of collisions
    involving 3 or more cars to the total number of collisions for each month of 2017.
    * Calculate the chi-square test statistic for testing whether a collision is more likely to involve 3 or more cars in January than in May.
    """
    questionSevenCol_list = [
        column_number for column_number in range(24, 29)
    ]
    questionSevenCol_list.append(0)


    def addMonthFilterByYearMyDF(focus_df=focusDataframer(questionSevenCol_list), date_column_label='DATE', bring_all_records_for='2017'):
        """
        REQUIRES A DATAFRAME, COLUMN NAME THAT HAS DATE OF EACH RECORD, 
        ADDS A COLUMN BY NAME 'ye4r' WITH FOUR DIGIT YEAR 'YYYY' STRING AS VALUES
        """
        pass
        print()
        print('> > > dateFilterMyDataFrame() filtering all records by year : ' +
              bring_all_records_for)
    #     focus_df = focusDataframer()
        focus_df['ye4r'] = [
            date[-4:] for date in focus_df['DATE']
        ]
        filtered_by_year_df = focus_df.loc[focus_df['ye4r']
                                           == bring_all_records_for]
        filtered_by_year_df['m0nth'] = [date[:2]
                                        for date in filtered_by_year_df[date_column_label]]
        return filtered_by_year_df.drop(columns=[date_column_label, 'ye4r'])


    def monthNamesLister(months_in_numbers_list=np.arange(12)):
        """ RETURNS LIST OF FULL NAMES OF MONTHS GIVEN AS 01 OR 02 OR 03 ETC"""
        monthNames_list = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        monthsNumber_list = np.arange(12)
        monthNames_dict = dict(zip(monthsNumber_list, monthNames_list))
        out_list = [
            monthNames_dict[integerMonth]
            for integerMonth in list(months_in_numbers_list)
        ]
        return out_list


    def finalDataFram3r(focusAddMonth_df=addMonthFilterByYearMyDF(bring_all_records_for='2017')):
        vehiclesOnly_df = focusAddMonth_df.groupby('m0nth').count()
        gr0uped_df = vehiclesOnly_df.copy()
        collusionTotal2017_num = focusAddMonth_df.count()[-1]
        vehicleColumns_list = list(gr0uped_df.columns)
        gr0uped_df['singleCarOnly'] = [
            int(first - second)
            for first, second
            in zip(gr0uped_df[vehicleColumns_list[0]], gr0uped_df[vehicleColumns_list[1]])
        ]
        gr0uped_df['twoCarsOnly'] = [
            int(first - second)
            for first, second
            in zip(gr0uped_df[vehicleColumns_list[1]], gr0uped_df[vehicleColumns_list[2]])
        ]
        # at least one vehicle involved
        gr0uped_df['numTotalMonthly'] = [
            int(minimum_one_car)
            for minimum_one_car
            in gr0uped_df[vehicleColumns_list[0]]
        ]
        final_df = gr0uped_df[['singleCarOnly',
                               'twoCarsOnly',
                               'numTotalMonthly']]
        final_df['threeCarsAndMore'] = [int(total) - int(single + double) for total, single, double in zip(final_df['numTotalMonthly'], final_df['singleCarOnly'], final_df['twoCarsOnly'])
                                        ]
        return final_df.drop(columns=['singleCarOnly', 'twoCarsOnly'])

    # Q-7: Do winter driving conditions lead to more multi-car collisions?
    # * Compute the rate of multi car collisions as the proportion of the number of collisions involving 3 or more cars to the total number of collisions for each month of 2017.


    def fin4lPerctgDataFramer(
        final_df=finalDataFram3r(
            focusAddMonth_df=addMonthFilterByYearMyDF(
                bring_all_records_for='2017'))):

        percentage_df = final_df.copy()
        percentage_df['percMonthly'] = [
            float(threecars/totalnum*100).__round__(2)
            for threecars, totalnum
            in zip(percentage_df['threeCarsAndMore'], percentage_df['numTotalMonthly'])
        ]

        return percentage_df.drop(columns='threeCarsAndMore')


    def bubbleMultiCarInv(
        multiCarInvPerc_df=fin4lPerctgDataFramer()
    ):
        """
        CREATES A BUBBLE CHART WITH BUBBLE SIZE PROPORTIONATE TO MONTHLY COLLUSION COUNT
        """
        plt.figure(figsize=(18, 13))
        plt.title('BUBBLE CHART: ZIP CODE VS COLLUSION-PER-SQ-KM-2017')
        plt.title('NYC Collusion Dataset-2017')
        plt.ylabel('Monthly Three or More Vehicle Involvement Percentage')
        plt.xlabel('Months')
        perc_list = multiCarInvPerc_df['percMonthly']
        num4size_list = multiCarInvPerc_df['numTotalMonthly']
        m0nthFancy_l1st = monthNamesLister()
        monthFancy_list = [
            month + ': {:,}'.format(numCollusion)
            for month, numCollusion in zip(m0nthFancy_l1st, num4size_list)
        ]

        for month, percentage, numTotal in zip(m0nthFancy_l1st, perc_list, num4size_list):
            lab3l = str(month)
            plt.scatter(
                month, percentage,
                s=3000*numTotal/num4size_list.max(),
                marker='o', edgecolors='black', alpha=0.50,
                label=lab3l
            )
        plt.legend(loc='best')
        fileName2save = 'Q7_MULTI-CAR-COLLUSION-MONTHLY-2017.png'
        plt.savefig(fileName2save)
        plt.xticks(rotation = 45)
        plt.show()
        return None


    bubbleMultiCarInv()

    return None
# solution07aof08()


def solution08of08():
    """
    # Q-8: We can use collision locations to estimate the areas of the zip code regions. (Assumption) Represent each as an ellipse with semi-axes given by a single standard deviation of the longitude and latitude. For collisions in 2017, estimate the number of collisions per square kilometer of each zip code region. Considering zipcodes with at least 1000 collisions, report the greatest value for collisions per square kilometer. Note: Some entries may have invalid or incorrect (latitude, longitude) coordinates. Drop any values that are invalid or seem unreasonable for New York City.
    """
    welcome_message = '> > > solution08of08() ... running...'
    print(welcome_message)
    def getRidOfZerosFromLatLon(yearFiltered_df = dateFilterMyDataFrame(focus_df = focusDataframer(), bring_all_records_for = '2017')):
        """
        REMOVES RECORDS WITH ZERO IN EITHER LATITUDE OR LONGITUDE FROM DATAFRAME
        SINCE NEITHER LAT-ZERO NOR LON-ZERO IS IN NYC
        """
        yf_df = yearFiltered_df.copy()
        yfd_df = yf_df.dropna().loc[:, (yf_df == 0).any()]
        yfd_df['zipCod3'] = [
            yf_df['ZIP CODE'][ind3x]
            for ind3x in list(yfd_df.index)
        ]
        return yfd_df

    def zipCodevsColSqKm_df(yearFilteredDroppedGrouped_df = getRidOfZerosFromLatLon()):
        """
        RETURNS A DATAFRAME ZIP CODE VS COLLUSION CT
        """
        yfdgCount_df = yearFilteredDroppedGrouped_df.groupby('zipCod3').count()  #YearFiltered_Dropped_GROUPED_DF
        yfdgCount_df['collusionCount'] = yfdgCount_df['LATITUDE']
        yfdgCount_df1 = yfdgCount_df.drop(columns=['LATITUDE','LONGITUDE'])
        yfdgCount_df2 = yfdgCount_df1.sort_values('collusionCount', ascending = False)
        zipCollusion_df = yfdgCount_df2.loc[yfdgCount_df2['collusionCount'] > 1000]
        return zipCollusion_df

    def standardDeviationZIPLatLan(yearFiltered_df = dateFilterMyDataFrame(focus_df = focusDataframer(), bring_all_records_for = '2017')):
    #     dropZeroLat_df = yearFiltered_df.loc[yearFiltered_df['LATITUDE'] == 0]
        print(yearFiltered_df.describe())
    #     print(dropZeroLat_df.describe())
        print('> > > standardDeviationZIPLatLan() calculated std of LAT and LAN: ')
        return yearFiltered_df.groupby('ZIP CODE').std()

    def zipCodeAreaSqKm(yearFilteredDroppedGrouped_df= standardDeviationZIPLatLan()):
        yfdg_df = yearFilteredDroppedGrouped_df.copy() 
        #YearFiltered_Dropped_GROUPED_DF
        PI = 3.14
        yfdg_df['are4'] = [
            float(PI/4*a *69*b*55)  for a,b in zip(yfdg_df['LATITUDE'], yfdg_df['LONGITUDE'])
        ]

        final_df = yfdg_df.drop(columns=['LATITUDE', 'LONGITUDE'])
        # .loc[yfdg_df['are4'] >= 1000]
        calculatedTotalArea = final_df.sort_values(by='are4', ascending=False).sum() #752,000 km2
        nyc_totalArea = 783800
        realization_factor = calculatedTotalArea/nyc_totalArea
        print('> > > zipCodevsColSqKm_df calculated estimated area as 752 thousand km2')
        print('> > >                   NYC total area actually is 783 thousand 800 km2')
        print('> > >                   realization factor has been used to adjust nominal to real values')
        print('> > >                   ',realization_factor)
        final_df['ar3a'] = [
            float(area*realization_factor)
            for area in final_df['are4']
        ]
        zipArea_df = final_df.drop(columns='are4').T
        return zipArea_df


    def zipCodeColPerSqKm(zipCollusion_df= zipCodevsColSqKm_df(), zipArea_df=zipCodeAreaSqKm()):
        """
        RETURNS ZIP CODE VS COLLUSION-PER-SQ-KM2 STORED IN A DATAFRAME
        """
        zipCollusion_df['AR3A'] = [
            zipArea_df[zipCode][0] for zipCode in list(zipCollusion_df.index)
        ]
        # zipArea_df 
        zipCollusion_df['colPerSqKm'] = [
            collusion/area
            for collusion, area in zip(zipCollusion_df['collusionCount'], zipCollusion_df['AR3A'])
        ]
        zipColSqKm_df= zipCollusion_df.sort_values(by='colPerSqKm', ascending=False)[[
            'colPerSqKm'
        ]]
        return zipColSqKm_df

    def answer08of08(zipColSqKm_df = zipCodeColPerSqKm()):
        z1p = zipColSqKm_df['colPerSqKm'].index[0]
        colMax = zipColSqKm_df['colPerSqKm'][0]
        print('> > > q8finalAnswer() for highest collusion per area density is {:,.0f} col/sq-km'.format(colMax))
        print('> > > > > > > > > > > while ZIP code with highest density above is ' + z1p)
        return None
    answer08of08()

    def bubbleZipCodeVsColPerSqKm(zipColSqKm_df = zipCodeColPerSqKm(), bubbleCount = 10):
        """
        PLOTS A BUBBLE CHART WITH BUBBLE SIZE PROPORTIONATE TO COLLUSION COUNT PER ZIP CODE
        """
        z1p_list = zipColSqKm_df['colPerSqKm'].index[:bubbleCount]
        colMax_list = zipColSqKm_df['colPerSqKm'][:bubbleCount]
        plt.figure(figsize=(22, 15))
        plt.title('BUBBLE CHART: ZIP CODE VS COLLUSION-PER-SQ-KM-2017')
        plt.ylabel('Collusion Per Sq-Km for 2017')
        plt.xlabel('ZIP Code (NYPD collusion dataset)')
        plt.xticks(rotation=45)
        for ZIPCODE,COLSQKM in zip(z1p_list,colMax_list):
            plt.scatter(ZIPCODE,COLSQKM, s=5000*COLSQKM/colMax_list.max(), marker='o', edgecolors='black' , alpha = 0.50,label = str(ZIPCODE))
        plt.legend(loc='best')
        fileName2save = 'Q8_ZIPCODE_COL-SQ-KM-2017.png'
        plt.savefig(fileName2save)
        plt.show()

    bubbleZipCodeVsColPerSqKm()

    return None

# solution08of08()
 
def main():
    # solution08of08()
    
    return None


if __name__ == '__main__':
    main()
