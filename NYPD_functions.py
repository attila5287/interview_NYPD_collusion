import os
import pandas as pd
import numpy as np
from generate_color4_plot import (
    generateRandomDark_c0lor,
    generateRandomLight_c0lor,
    generateRandomMatplotlib_c0lor
    )
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
def dataFram3NYC(FILE_NAME = 'NYPD_Motor_Vehicle_Collisions.csv', use_parent_dir = False):
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
    PATH_COMPILED = os.path.join( BASE_DIR, STATIC_FOLDER, FILE_NAME)
    print('\n> > > > dataFram3NYC() now reading csv with file path below')
    print('> > > ', PATH_COMPILED)
    print()
    _df = pd.read_csv(PATH_COMPILED, encoding = "iso-8859-1", low_memory = False)
    col_list = list(_df.columns)
    col_ind3x = np.arange(len(col_list))
    col_dict = dict(zip( col_ind3x, col_list))
    print(col_dict)
    return _df
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
def focusDataframer(columns2keep_list = [0,3,4,5]):
    """
    MODIFIED DATA FRAMER, PRINTS OUT ALL COLUMN INDEX/NAMES, 
    REQUIRES A LIST OF REQUESTED COLUMN INDEXES AS INTEGERS
    RETURNS A DATAFRAME WITH ONLY USER-SELECTED COLUMNS
    """
    pass
    raw_df = dataFram3NYC(use_parent_dir=True)
    col_list = list(raw_df.columns)
    col_ind3x = np.arange(len(col_list))
    col_dict = dict(zip( col_ind3x, col_list))
    col_final = []
    for index in columns2keep_list:
        add_this_column = col_dict[index]
        col_final.append(add_this_column)
    col_final = [
        str(index)
        for index in col_final
    ]
    _df = raw_df[
        col_final
    ]
    print()
    print('> > > focusDataframer(columns2keep_list = [indexNo_as_int00, indexNo_as_int01]')
    print('> > >                * selected columns:')
    for selected_col,keepthis in zip(columns2keep_list,col_final):
        print('> > >                                  ', selected_col,' keeps ', keepthis)
    return _df
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
def dateFilterMyDataFrame(focus_df = focusDataframer(), bring_all_records_for = '2016'):
    """
    REQUIRES A DATAFRAME WITH A TARGET FIELD NAMED 'DATE',
    FILTERS DATE BY FOUR DIGIT YEAR 'YYYY' STRING INPUT
    """
    pass
    print()
    print('> > > dateFilterMyDataFrame() filtering all records by year : ' + bring_all_records_for)
#     focus_df = focusDataframer()
    focus_df['dat3'] = [
        date[-4:] for date in focus_df['DATE']
    ]
    filtered_by_date_df = focus_df.loc[focus_df['dat3'] == bring_all_records_for]
    return filtered_by_date_df.drop(columns = 'DATE')
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
def addMonthFilterByYearMyDF(focus_df = focusDataframer(), date_column_label = 'DATE', bring_all_records_for = '2016'):
    """
    REQUIRES A DATAFRAME, COLUMN NAME THAT HAS DATE OF EACH RECORD, 
    ADDS A COLUMN BY NAME 'ye4r' WITH FOUR DIGIT YEAR 'YYYY' STRING AS VALUES
    """
    pass
    print()
    print('> > > dateFilterMyDataFrame() filtering all records by year : ' + bring_all_records_for)
#     focus_df = focusDataframer()
    focus_df['ye4r'] = [
        date[-4:] for date in focus_df['DATE']
    ]
    filtered_by_year_df = focus_df.loc[focus_df['ye4r'] == bring_all_records_for]
    filtered_by_year_df['m0nth'] = [date[:2] for date in filtered_by_year_df['DATE']]
    return filtered_by_year_df.drop(columns = ['DATE', 'ye4r'])
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
def monthNamesLister(months_in_numbers_list = np.arange(12)):

    """ RETURNS LIST OF FULL NAMES OF MONTHS GIVEN AS 01 OR 02 OR 03 ETC"""
    monthNames_list= [
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
# -----  -----  -----  -----  -----  -----  -----  -----  -----  -----  
