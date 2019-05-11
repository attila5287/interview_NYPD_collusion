import os
import pandas as pd

def dataFrameNYC(FILE_NAME = 'NYPD_Motor_Vehicle_Collisions.csv'):
    """
    READS ~1,5 MILLION ROW CSV FILE, RETURNS MASTER DATAFRAME
    """
    BASE_DIR = os.getcwd()
    STATIC_FOLDER = '../'
    # FILE_NAME = 'NYPD_Motor_Vehicle_Collisions.csv'
    PATH_COMPILED = os.path.join(BASE_DIR, STATIC_FOLDER, FILE_NAME)
    print(PATH_COMPILED)
    print()
    _df = pd.read_csv(PATH_COMPILED, encoding = "iso-8859-1", low_memory = False)
    column_list = _df.columns
    print(column_list)
    return _df



# dataFrameNYC()