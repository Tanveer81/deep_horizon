"""
Read the raw data and generate the data set
"""
import os
from datetime import timedelta, datetime

import cdflib
import numpy as np
import pandas as pd
from dateutil.parser import parse

COLUMNS = {
    'CHANNEL': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
    'POS': ['x', 'y', 'z'],
    'RDIST': ['rdist'],
    'OMNI': [
        'AE_index', 'SYM-H_index', 'F107', 'BimfxGSE', 'BimfyGSE', 'BimfzGSE',
        'VxSW_GSE', 'VySW_GSE', 'VzSW_GSE', 'NpSW', 'Pdyn', 'Temp'
    ],
    'DATETIME': ['DateTime']
}
RAPID = COLUMNS['CHANNEL'] + COLUMNS['POS'] + COLUMNS['RDIST']
OMNI = COLUMNS['OMNI']
START_YEAR = 2001
END_YEAR = 2019
RAP_PATH = 'C4_CP_RAP_HSPCT/'
POS_PATH = 'C4_CP_AUX_POSGSE_1M/'
RAP_TIME_VAR = 'Time_tags__C4_CP_RAP_HSPCT'
RAP_FLUX_VAR = 'Proton_Dif_flux__C4_CP_RAP_HSPCT'
POS_TIME_VAR = 'time_tags__C4_CP_AUX_POSGSE_1M'
POS_POS_VAR = 'sc_r_xyz_gse__C4_CP_AUX_POSGSE_1M'
OMNI_PATH = 'OMNI/'
R_EARTH = 6371.1

if __name__ == '__main__':
    # RAPID
    # setting range
    file_dates = pd.date_range(datetime(START_YEAR, 1, 1),
                               datetime(END_YEAR, 12, 31),
                               freq='D').astype(str).str.replace('-', '')
    # search for valid filed
    rapid_list = {f.split('_')[5]: RAP_PATH + f for f in os.listdir(RAP_PATH) if f[-3:] == 'cdf'}
    pos_list = {f.split('_')[6]: POS_PATH + f for f in os.listdir(POS_PATH) if f[-3:] == 'cdf'}

    # traverse all dates
    cluster = []
    for file_date in file_dates:
        if file_date[6:] == '01':
            print(file_date[:6])  # print progress
        # read proton intensities
        rapid_file = rapid_list.get(file_date, None)
        if rapid_file:
            rapid_cdf = cdflib.CDF(rapid_file)
            # get timestamp
            try:
                rapid_time = rapid_cdf.varget(RAP_TIME_VAR)
            except ValueError:
                pass
            else:
                if len(rapid_time) > 1 or rapid_time[0] != -1.e-31:
                    offset_rap = (parse('1970-01-01') - parse('0001-01-01') +
                                  timedelta(days=366)).total_seconds() * 1000
                    rapid_time = pd.DataFrame(pd.to_datetime((rapid_time - offset_rap) * 1000000),
                                              columns=['DateTime'])
                    rapid_time['DateTime'] = rapid_time['DateTime'].dt.floor('s')
                    rapid_time['DateTime_min'] = rapid_time['DateTime'].dt.floor('min')  # minute
                    # get intensities
                    intensities = rapid_cdf.varget(RAP_FLUX_VAR).astype('float64')
                    intensities = np.array(intensities).reshape((-1, 8))[:, :7]
                    intensities = pd.DataFrame(intensities, columns=[f'p{i}' for i in range(1, 8)])
                    intensities[(intensities < 0) | (intensities > 1e8)] = np.nan
                    # merge intensities and time stamp
                    flux_with_time = pd.concat((intensities, rapid_time), axis=1)
                    # read position
                    pos_file = pos_list.get(file_date, None)
                    if pos_file:
                        pos_cdf = cdflib.CDF(pos_file)
                        # get timestamp
                        pos_time = pos_cdf.varget(POS_TIME_VAR)
                        offset_pos = (parse('1970-01-01') - parse('0001-01-01') +
                                      timedelta(days=366)).total_seconds() * 1000
                        pos_time = pd.DataFrame(pd.to_datetime((pos_time - offset_pos) * 1000000),
                                                columns=['DateTime_min'])
                        try:
                            positions = pos_cdf.varget(POS_POS_VAR).astype('float64')
                        except ValueError:
                            pass
                        else:
                            positions = pd.DataFrame(positions, columns=['x', 'y', 'z'])
                            positions /= R_EARTH
                            positions['rdist'] = pow(positions['x'] ** 2 +
                                                     positions['y'] ** 2 +
                                                     positions['z'] ** 2, 0.5)
                            posgse = pd.concat((positions, pos_time), axis=1)
                            merged = pd.merge(flux_with_time, posgse, on='DateTime_min')
                            # aggregate
                            merged.drop(['DateTime'], axis=1, inplace=True)
                            merged = merged.groupby('DateTime_min', as_index=False).mean()
                            merged[['DateTime']] = merged.loc[:, ['DateTime_min']]
                            # append the day to the data set
                            cluster.append(merged)
    cluster = pd.concat(cluster, axis=0)

    # OMNI
    years = np.arange(START_YEAR, END_YEAR + 1)
    omni_high = []
    for year in years:
        data = pd.read_csv(f"{OMNI_PATH}omni_min_{year}.lst.txt", delimiter=r'\s+')
        omni_high.append(data)
    high = pd.concat(omni_high, axis=0)
    omni_low = pd.read_csv(f"{OMNI_PATH}omni_hour.lst.txt", delimiter=r'\s+')
    omni = pd.merge(high, omni_low, on=['Year', 'DOY', 'Hour']).astype('float64')
    omni['Pdyn'] = omni['NpSW'] * (omni['VxSW_GSE'] * omni['VxSW_GSE'] +
                                   omni['VySW_GSE'] * omni['VySW_GSE'] +
                                   omni['VzSW_GSE'] * omni['VzSW_GSE']) * 1.67e-6
    omni['DateTime'] = pd.to_datetime(omni['Year'].astype(int).astype(str) + '-' +
                                      omni['DOY'].astype(int).astype(str) + '-' +
                                      omni['Hour'].astype(int).astype(str) + '-' +
                                      omni['Minute'].astype(int).astype(str), format='%Y-%j-%H-%M')

    for column in ['AE_index', 'F107', 'BimfxGSE', 'BimfyGSE', 'BimfzGSE', 'VxSW_GSE',
                   'VySW_GSE', 'VzSW_GSE', 'NpSW', 'Pdyn', 'Temp']:
        omni.loc[:, column] = omni.loc[:, column].replace(omni.loc[:, column].max(), np.nan)
    omni = omni.dropna()

    # merge
    data = cluster.merge(omni, on='DateTime', how='outer')
    data = data[RAPID + OMNI + COLUMNS['DATETIME']]
    data.to_hdf('RAPID_OMNI_ML_023_raw.h5', key='df')
