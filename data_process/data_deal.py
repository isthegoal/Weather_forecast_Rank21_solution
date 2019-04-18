# coding=utf-8
'''
  这里对提取出来的数据，进行预处理：
       *阈值限制，超过这个阈值的都设置为空
       *均值或者线性插值填充，根据数据画图情况进行填充
'''
import pandas as pd
import numpy as np
import h5py
import pickle
import time
from sklearn import preprocessing
from datetime import datetime
import datetime
import netCDF4 as nc
import sys
import glob
import csv
import numpy as np
import pandas as pd
import time
import datetime

def do_data_process():
    print('3333')
    #果然加了index_col=0后，就不会多出一行了
    the_ruitu_df = pd.read_csv('./data/ruitu_all_time_data.csv',index_col=0)
    the_gc_df=pd.read_csv('./data/gc_all_time_data.csv',index_col=0)
    #print(the_ruitu_df)

    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    ruitu_columns = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M',
                     'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M',
                     'wspd975_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
                     'Q700_M', 'Q500_M']
    print('---------------   限制阈值（超出阈值的直接限定为null）   ---------------')
    print('---  对gc数据进行限定处理  ---')
    the_gc_df.loc[the_gc_df['psur_obs'] > 1100, 'psur_obs'] = np.nan
    the_gc_df.loc[the_gc_df['psur_obs'] < 850, 'psur_obs'] = np.nan
    the_gc_df.loc[the_gc_df['t2m_obs'] > 55, 't2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['t2m_obs'] < -40, 't2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['q2m_obs'] > 30, 'q2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['q2m_obs'] < 0, 'q2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['rh2m_obs'] > 100, 'rh2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['rh2m_obs'] < 0, 'rh2m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['w10m_obs'] > 30, 'w10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['w10m_obs'] < 0, 'w10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['d10m_obs'] > 360, 'd10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['d10m_obs'] < 0, 'd10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['u10m_obs'] > 30, 'u10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['u10m_obs'] < -30, 'u10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['v10m_obs'] > 30, 'v10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['v10m_obs'] < -30, 'v10m_obs'] = np.nan
    the_gc_df.loc[the_gc_df['RAIN_obs'] > 400, 'RAIN_obs'] = np.nan
    the_gc_df.loc[the_gc_df['RAIN_obs'] < 0, 'RAIN_obs'] = np.nan
    print('---  对ruitu数据进行限定处理  ---')
    the_ruitu_df.loc[the_ruitu_df['psfc_M'] > 1100, 'psfc_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['psfc_M'] < 850, 'psfc_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['t2m_M'] > 55, 't2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['t2m_M'] < -40, 't2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['q2m_M'] > 30, 'q2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['q2m_M'] < 0, 'q2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['rh2m_M'] > 100, 'rh2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['rh2m_M'] < 0, 'rh2m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['w10m_M'] > 30, 'w10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['w10m_M'] < 0, 'w10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['d10m_M'] > 360, 'd10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['d10m_M'] < 0, 'd10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['u10m_M'] > -30, 'u10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['u10m_M'] < 30, 'u10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['v10m_M'] > -30, 'v10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['v10m_M'] < 30, 'v10m_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['SWD_M'] > 1500, 'SWD_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['SWD_M'] < 0, 'SWD_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['GLW_M'] > 800, 'GLW_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['GLW_M'] < 0, 'GLW_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['HFX_M'] > 1000, 'HFX_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['HFX_M'] < -400, 'HFX_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['LH_M'] > 1000, 'LH_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['LH_M'] < -1000, 'LH_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['RAIN_M'] > 400, 'RAIN_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['RAIN_M'] < 0, 'RAIN_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['PBLH_M'] > 6000, 'PBLH_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['PBLH_M'] < 0, 'PBLH_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC975_M'] > 45, 'TC975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC975_M'] < -50, 'TC975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC925_M'] > 45, 'TC925_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC925_M'] < -50, 'TC925_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC850_M'] > -55, 'TC850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC850_M'] < 40, 'TC850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC700_M'] > 35, 'TC700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC700_M'] < -60, 'TC700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC500_M'] > 30, 'TC500_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['TC500_M'] < -70, 'TC500_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd975_M'] > 60, 'wspd975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd975_M'] < 0, 'wspd975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd925_M'] > 70, 'wspd925_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd925_M'] < 0, 'wspd925_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd850_M'] > 80, 'wspd850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd850_M'] < 0, 'wspd850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd700_M'] > 90, 'wspd700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd700_M'] < 0, 'wspd700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd500_M'] > 100, 'wspd500_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['wspd500_M'] < 0, 'wspd500_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q975_M'] > 30, 'Q975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q975_M'] < 0, 'Q975_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q925_M'] > 30, 'psfc_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q925_M'] < 0, 'psfc_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q850_M'] > 30, 'Q850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q850_M'] < 0, 'Q850_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q700_M'] > 25, 'Q700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q700_M'] < 0, 'Q700_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q500_M'] > 25, 'Q500_M'] = np.nan
    the_ruitu_df.loc[the_ruitu_df['Q500_M'] < 0, 'Q500_M'] = np.nan

    print('---------------   缺失值使用均值填充或者插值法补全（对于未知的要预测的位置也补全吧，反正到时候也不用，空位置而已）   ---------------')
    #这里基线首先只用线性插值进行填充，填充之前设置排序方式为按站点进行组内排序，这样内部排序后 再进行时序上的插值更为合适。
    print('------    使用插值法进行限定，限定前先将同一站点的数据合在一起，因为这样的插入更加准确    ------')
    print('---  对观测数据进行插值处理  ---')
    the_gc_df_sort=[]
    j=0
    def gc_hebing(i):
        i=i.sort_values(by='time')
        for index, row in i.iterrows():
            the_gc_df_sort.append(row)
    the_gc_df.groupby('the_station').apply(gc_hebing)
    the_gc_df_sorted=pd.DataFrame(the_gc_df_sort)
    #上面的apply方式总是多出来一次apply使用，所以我们这里来一次去重
    the_gc_df_sorted=the_gc_df_sorted.drop_duplicates(['time','the_station'],keep = "first")
    the_gc_df_sorted=the_gc_df_sorted.interpolate()
    print('---  对睿图数据进行插值处理  ---')
    the_ruitu_df_sort=[]
    def ruitu_hebing(i):
        i=i.sort_values(by='time')
        #print('经过时间排序',i)
        for index, row in i.iterrows():
            the_ruitu_df_sort.append(row)
    the_ruitu_df.groupby('the_station').apply(ruitu_hebing)
    the_ruitu_df_sorted=pd.DataFrame(the_ruitu_df_sort)
    the_ruitu_df_sorted=the_ruitu_df_sorted.drop_duplicates(['time','the_station'],keep = "first")
    the_ruitu_df_sorted=the_ruitu_df_sorted.interpolate()

    print('---------------    将预处理的结果进行保存     ---------------')
    the_gc_df_sorted.to_csv('./data_process/process_data/do_fill_na_gc.csv')
    the_ruitu_df_sorted.to_csv('./data_process/process_data/do_fill_na_ruitu.csv')

if __name__=='__main__':
    do_data_process()

