# -*- coding:utf-8 -*-
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

print('-------   每天需要修改的参数   【开始预测的时间点  天数   文件地址   检查提交结果】--------')
the_pred_file='../data/ai_challenger_wf2018_testb3_20180829-20181030.nc'
the_file_day_number=63
the_strat_pred_time='2018/10/30 03:00:00'


def nc_to_csv_ruitu():
    print('-------  加载需要解析的文件（官方实时提供的文件）从中获取所有睿图信息  -------')
    the_testB_data = nc.Dataset(the_pred_file)
    ruitu_columns=['psfc_M','t2m_M','q2m_M','rh2m_M','w10m_M','d10m_M','u10m_M','v10m_M','SWD_M','GLW_M','HFX_M','LH_M','RAIN_M','PBLH_M','TC975_M','TC925_M','TC850_M','TC700_M','TC500_M','wspd975_M','wspd925_M','wspd850_M','wspd700_M','wspd500_M','Q975_M','Q925_M','Q850_M','Q700_M','Q500_M']
    the_all_time=[]
    the_big_ruitu_table = []
    print('------- 现在举例对48天的测试集2进行处理(48, 37, 10) -----')
    #对于the_testB_data
    for i in range(0, the_file_day_number):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0,37):
            #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            #第一步加入时间点
            the_ori_date=the_testB_data['date'][i]
            the_formated_date=pd.to_datetime(str(the_ori_date)[:-2],format='%Y%m%d%H')
            the_real_data=the_formated_date+datetime.timedelta(hours=j)

            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            #print(the_real_data)

            #第二步加入站点信息
            the_station_name=['90001', '90002', '90003', '90004' ,'90005' ,'90006' ,'90007' ,'90008' ,'90009', '90010']
            for k in range(0,10):

                the_hang = []#不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                #第三步加入每个属性的信息
                for t in  ruitu_columns:
                    the_hang.append(np.array(the_testB_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_ruitu_table.append(the_hang)

            #print('要加入的一行是',the_big_ruitu_table)

    the_ruitu_df=pd.DataFrame(the_big_ruitu_table,columns=['time','the_station','psfc_M','t2m_M','q2m_M','rh2m_M','w10m_M','d10m_M','u10m_M','v10m_M','SWD_M','GLW_M','HFX_M','LH_M','RAIN_M','PBLH_M','TC975_M','TC925_M','TC850_M','TC700_M','TC500_M','wspd975_M','wspd925_M','wspd850_M','wspd700_M','wspd500_M','Q975_M','Q925_M','Q850_M','Q700_M','Q500_M'])
    the_ruitu_df.to_csv('./for_submit_do_data/ruitu_all_time_data.csv')


def nc_to_csv_gc():
    the_testB_data = nc.Dataset(the_pred_file)
    guance_columns=['psur_obs','t2m_obs','q2m_obs','rh2m_obs','w10m_obs','d10m_obs','u10m_obs','v10m_obs','RAIN_obs']
    print('---------------------------  提取出观测信息  -------------------------------')
    the_big_gc_table=[]
    the_all_time=[]
    print('-- 对有48天的测试集2进行处理(48, 37, 10) --')
    #对于the_testB_data
    for i in range(0, the_file_day_number):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0,37):
            #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            #第一步加入时间点
            the_ori_date=the_testB_data['date'][i]
            the_formated_date=pd.to_datetime(str(the_ori_date)[:-2],format='%Y%m%d%H')
            the_real_data=the_formated_date+datetime.timedelta(hours=j)

            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            #第二步加入站点信息
            the_station_name=['90001', '90002', '90003', '90004' ,'90005' ,'90006' ,'90007' ,'90008' ,'90009', '90010']
            for k in range(0,10):
                the_hang = []#不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                #第三步加入每个属性的信息
                for t in  guance_columns:
                    the_hang.append(np.array(the_testB_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_gc_table.append(the_hang)

    the_gc_df=pd.DataFrame(the_big_gc_table,columns=['time','the_station','psur_obs','t2m_obs','q2m_obs','rh2m_obs','w10m_obs','d10m_obs','u10m_obs','v10m_obs','RAIN_obs'])
    the_gc_df.to_csv('./for_submit_do_data/gc_all_time_data.csv')

def do_data_process():
    #果然加了index_col=0后，就不会多出一行了
    the_ruitu_df = pd.read_csv('./for_submit_do_data/ruitu_all_time_data.csv',index_col=0)
    the_gc_df=pd.read_csv('./for_submit_do_data/gc_all_time_data.csv',index_col=0)
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
        for index, row in i.iterrows():
            the_ruitu_df_sort.append(row)
    the_ruitu_df.groupby('the_station').apply(ruitu_hebing)
    the_ruitu_df_sorted=pd.DataFrame(the_ruitu_df_sort)
    the_ruitu_df_sorted=the_ruitu_df_sorted.drop_duplicates(['time','the_station'],keep = "first")
    the_ruitu_df_sorted=the_ruitu_df_sorted.interpolate()

    print('---------------    将预处理的结果进行保存     ---------------')
    the_gc_df_sorted.to_csv('./for_submit_do_data/do_fill_na_gc.csv')
    the_ruitu_df_sorted.to_csv('./for_submit_do_data/do_fill_na_ruitu.csv')


def gen_time_window_data():
    '''
    预定思路是根据要启动预测的时间点，找到这个时间点对应的滑动数据，所以本质上大滑动只需构造10条即可，小滑动需要构造340条
    '''
    #首先假定出要预测的时间点，正式赛第一个时间点应该是'2018/10/28 03:00:00'
    the_need_date_point=pd.to_datetime(the_strat_pred_time)
    print('---------------  小滑动，这里构造出10个站点要预测时刻的滑窗数据  -------------------')

    print('--  读取数据  --')
    ruitu_data=pd.read_csv('./for_submit_do_data/do_fill_na_ruitu.csv',index_col=0)
    gc_data=pd.read_csv('./for_submit_do_data/do_fill_na_gc.csv',index_col=0)
    #对日期转换下，不然认为是string型的
    ruitu_data['time'] = pd.to_datetime(ruitu_data['time'])
    gc_data['time'] = pd.to_datetime(gc_data['time'])
    print('--  对于单个时刻点，开始对每个站点构建小型滑动  --')

    the_big_tog_data = []
    for station in [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009, 90010]:
        # 对于每个站点，分别进行下面的过程
        print('---------   获取这个时间点的前40个小时的观测数据       ----------')
        the_before_start_slide_date = the_need_date_point + datetime.timedelta(hours=-120)
        the_first_range_data = gc_data[
            (gc_data['the_station'] == station) & (gc_data['time'] >= the_before_start_slide_date) & (
                        gc_data['time'] < the_need_date_point)]
        print('检查下是否  发生时间顺序错乱',the_first_range_data['time'])
        #print(the_first_range_data)
        the_first_range_data.pop('time')
        the_first_range_data.pop('the_station')
        #print('需要的第一个列名列表： ', the_first_range_data.columns)
        first_data = list(np.hstack(the_first_range_data.values))
        # print('前144个小时对应的数据维度为：',len(first_data))
        # guance_before_40_data.append(first_data)

        print('---------   获取这个时间点的开始的37个小时睿图的数据   ----------')
        the_last_slide_date = the_need_date_point + datetime.timedelta(hours=36)
        the_secong_range_data = ruitu_data[
            (ruitu_data['the_station'] == station) & (ruitu_data['time'] >= the_need_date_point) & (
                        ruitu_data['time'] <= the_last_slide_date)]
        # print('检查下是否  发生时间顺序错乱',the_first_range_data['utc_time'])
        the_secong_range_data.pop('time')
        the_secong_range_data.pop('the_station')
        #print('需要的第二个列名列表： ', the_secong_range_data.columns)
        second_data = list(np.hstack(the_secong_range_data.values))
        # print('前144个小时对应的数据维度为：',len(first_data))
        # ruitu_hou_37_data.append(second_data)
        print('---------   获取这个时间点的开始的37个小时观测的数据   ----------')
        the_last_slide_date = the_need_date_point + datetime.timedelta(hours=36)
        the_third_range_data = gc_data[
            (gc_data['the_station'] == station) & (gc_data['time'] >= the_need_date_point) & (
                        gc_data['time'] <= the_last_slide_date)]
        # print('检查下是否  发生时间顺序错乱',the_first_range_data['utc_time'])
        the_third_range_data.pop('time')
        the_third_range_data.pop('the_station')
        #print('需要的第三个列名列表： ', the_third_range_data.columns)
        third_data = list(np.hstack(the_third_range_data.values))
        # print('前144个小时对应的数据维度为：',len(first_data))
        # guance_hou_37_data.append(third_data)

        print('---------   进行信息的堆积（时间点、站点、前40个小时的观测数据、时间点开始的37个小时观测的数据、时间点开始的37个小时睿图的数据）   ----------')
        the_big_together = np.hstack((the_need_date_point, station, first_data, second_data, third_data))
        the_big_tog_data.append(list(the_big_together))
    last_big_dataset = pd.DataFrame(the_big_tog_data)

    last_big_dataset.to_csv('./for_submit_do_data/do__slice_data_da.csv')


def gen_time_window_data_gen_flag():

    print('---------------  对于产生的10行进行大型滑动（接下来10行滑动出340条数据来）  -------------------')
    last_big_dataset=pd.read_csv('./for_submit_do_data/do__slice_data_da.csv',index_col=0)
    print(len(last_big_dataset.columns))
    the_common_data = last_big_dataset.ix[:, 2:-297]
    # print(last_big_dataset.head())
    the_date = last_big_dataset.ix[:, 0]
    output = open('./for_submit_do_data/the_time.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(the_date, output)

    the_station_forma = list(last_big_dataset.ix[:, 1])
    # print('经过独热编码后的数据：',the_station_forma)
    # 单独获取后面34小时的观测点数据
    the_flag_data = last_big_dataset.ix[:, -297:]

    print('---------   分别分离出t2m_obs  rh2m_obs  w10m_obs  （位于观察序列的第2个、第4个、第5个）----------')
    t2m_obs_clo = []
    rh2m_obs_clo = []
    w10m_obs_clo = []
    number = 2487-8-32*9
    while (number <= 2487):
        t2m_obs_clo.append(str(number + 1))
        rh2m_obs_clo.append(str(number + 3))
        w10m_obs_clo.append(str(number + 4))
        number = number + 9

    t2m_all_data = []
    rh2m_all_data = []
    w10m_all_data = []



    print(
        '---------   对于每个变量构造新的行组合（用于构建时间属性的时间点、onehot站点标识[10]、前面的观察特征和3小时观察特征[3*9+9*40]、37小时睿图特征、启动时间特征、实际预测特征）  ----------')
    print('--------------     构造特征列名      ---------------')
    the_common_data_col = []
    ruitu_columns = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M',
                     'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M',
                     'wspd975_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
                     'Q700_M', 'Q500_M']
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    for i in range(0, 120):
        for j in guance_columns:
            the_common_data_col.append('guance_before_hour_' + str(40 - i) + '_' + j)
    for i in range(0, 37):
        for j in ruitu_columns:
            the_common_data_col.append('ruitu_future_hour_' + str(i) + '_' + j)
    for i in range(0, 4):
        for j in guance_columns:
            the_common_data_col.append('guance_future_hour_' + str(i) + '_' + j)
    the_common_data = pd.DataFrame(the_common_data.values, columns=the_common_data_col)
    print('总行数：', the_common_data.iloc[:, 0].size)



    print('-----  预测t2m_obs模型需要的数据  -----')
    for index, row in the_common_data.iterrows():

        print('--------------     构造时间特征      ---------------')
        # 算了，不想转移时序时间特征了，放在 feture_gen中的make_time_series_fea中了

        '''
          在每个前面都加入统计特征，这样替代后面的那种费时方式。
             构建的统计特征包括：
                 前前面9个指标的前12个小时、前24小时、前36小时的均值，最大值，最小值.三个指标均值的比值
                 共 3*3+3*3+3*3+2+2+2

          我们主要从the_common_data代表的前面360列数据中构造时序，其中

          '''

        print('--------------     构造统计特征      ---------------')
        the_before_12_hour_t2m_obs_list = ['guance_before_hour_' + str(i) + '_t2m_obs' for i in range(1, 12)]
        the_before_24_hour_t2m_obs_list = ['guance_before_hour_' + str(i) + '_t2m_obs' for i in range(1, 24)]
        the_before_36_hour_t2m_obs_list = ['guance_before_hour_' + str(i) + '_t2m_obs' for i in range(1, 36)]
        the_before_72_hour_t2m_obs_list = ['guance_before_hour_' + str(i) + '_t2m_obs' for i in range(1, 72)]
        the_before_108_hour_t2m_obs_list = ['guance_before_hour_' + str(i) + '_t2m_obs' for i in range(1, 108)]
        mean_t2m_12 = np.mean(the_common_data.loc[index, the_before_12_hour_t2m_obs_list])
        mean_t2m_24 = np.mean(the_common_data.loc[index, the_before_24_hour_t2m_obs_list])
        mean_t2m_36 = np.mean(the_common_data.loc[index, the_before_36_hour_t2m_obs_list])
        mean_t2m_72 = np.mean(the_common_data.loc[index, the_before_72_hour_t2m_obs_list])
        mean_t2m_108 = np.mean(the_common_data.loc[index, the_before_108_hour_t2m_obs_list])
        min_t2m_12 = np.min(the_common_data.loc[index, the_before_12_hour_t2m_obs_list])
        min_t2m_24 = np.min(the_common_data.loc[index, the_before_24_hour_t2m_obs_list])
        min_t2m_36 = np.min(the_common_data.loc[index, the_before_36_hour_t2m_obs_list])
        min_t2m_72 = np.min(the_common_data.loc[index, the_before_72_hour_t2m_obs_list])
        min_t2m_108 = np.min(the_common_data.loc[index, the_before_108_hour_t2m_obs_list])
        max_t2m_12 = np.max(the_common_data.loc[index, the_before_12_hour_t2m_obs_list])
        max_t2m_24 = np.max(the_common_data.loc[index, the_before_24_hour_t2m_obs_list])
        max_t2m_36 = np.max(the_common_data.loc[index, the_before_36_hour_t2m_obs_list])
        max_t2m_72 = np.max(the_common_data.loc[index, the_before_72_hour_t2m_obs_list])
        max_t2m_108 = np.max(the_common_data.loc[index, the_before_108_hour_t2m_obs_list])

        the_before_12_hour_rh2m_obs_list = ['guance_before_hour_' + str(i) + '_rh2m_obs' for i in range(1, 13)]
        the_before_24_hour_rh2m_obs_list = ['guance_before_hour_' + str(i) + '_rh2m_obs' for i in range(1, 25)]
        the_before_36_hour_rh2m_obs_list = ['guance_before_hour_' + str(i) + '_rh2m_obs' for i in range(1, 36)]
        the_before_72_hour_rh2m_obs_list = ['guance_before_hour_' + str(i) + '_rh2m_obs' for i in range(1, 72)]
        the_before_108_hour_rh2m_obs_list = ['guance_before_hour_' + str(i) + '_rh2m_obs' for i in range(1, 108)]
        mean_rh2m_12 = np.mean(the_common_data.loc[index, the_before_12_hour_rh2m_obs_list])
        mean_rh2m_24 = np.mean(the_common_data.loc[index, the_before_24_hour_rh2m_obs_list])
        mean_rh2m_36 = np.mean(the_common_data.loc[index, the_before_36_hour_rh2m_obs_list])
        mean_rh2m_72 = np.mean(the_common_data.loc[index, the_before_72_hour_rh2m_obs_list])
        mean_rh2m_108 = np.mean(the_common_data.loc[index, the_before_108_hour_rh2m_obs_list])
        min_rh2m_12 = np.min(the_common_data.loc[index, the_before_12_hour_rh2m_obs_list])
        min_rh2m_24 = np.min(the_common_data.loc[index, the_before_24_hour_rh2m_obs_list])
        min_rh2m_36 = np.min(the_common_data.loc[index, the_before_36_hour_rh2m_obs_list])
        min_rh2m_72 = np.min(the_common_data.loc[index, the_before_72_hour_rh2m_obs_list])
        min_rh2m_108 = np.min(the_common_data.loc[index, the_before_108_hour_rh2m_obs_list])
        max_rh2m_12 = np.max(the_common_data.loc[index, the_before_12_hour_rh2m_obs_list])
        max_rh2m_24 = np.max(the_common_data.loc[index, the_before_24_hour_rh2m_obs_list])
        max_rh2m_36 = np.max(the_common_data.loc[index, the_before_36_hour_rh2m_obs_list])
        max_rh2m_72 = np.max(the_common_data.loc[index, the_before_72_hour_rh2m_obs_list])
        max_rh2m_108 = np.max(the_common_data.loc[index, the_before_108_hour_rh2m_obs_list])

        the_before_12_hour_w10m_obs_list = ['guance_before_hour_' + str(i) + '_w10m_obs' for i in range(1, 13)]
        the_before_24_hour_w10m_obs_list = ['guance_before_hour_' + str(i) + '_w10m_obs' for i in range(1, 25)]
        the_before_36_hour_w10m_obs_list = ['guance_before_hour_' + str(i) + '_w10m_obs' for i in range(1, 36)]
        the_before_72_hour_w10m_obs_list = ['guance_before_hour_' + str(i) + '_w10m_obs' for i in range(1, 72)]
        the_before_108_hour_w10m_obs_list = ['guance_before_hour_' + str(i) + '_w10m_obs' for i in range(1, 108)]

        mean_w10m_12 = np.mean(the_common_data.loc[index, the_before_12_hour_w10m_obs_list])
        mean_w10m_24 = np.mean(the_common_data.loc[index, the_before_24_hour_w10m_obs_list])
        mean_w10m_36 = np.mean(the_common_data.loc[index, the_before_36_hour_w10m_obs_list])
        mean_w10m_72 = np.mean(the_common_data.loc[index, the_before_72_hour_w10m_obs_list])
        mean_w10m_108 = np.mean(the_common_data.loc[index, the_before_108_hour_w10m_obs_list])
        min_w10m_12 = np.min(the_common_data.loc[index, the_before_12_hour_w10m_obs_list])
        min_w10m_24 = np.min(the_common_data.loc[index, the_before_24_hour_w10m_obs_list])
        min_w10m_36 = np.min(the_common_data.loc[index, the_before_36_hour_w10m_obs_list])
        min_w10m_72 = np.min(the_common_data.loc[index, the_before_72_hour_w10m_obs_list])
        min_w10m_108 = np.min(the_common_data.loc[index, the_before_108_hour_w10m_obs_list])
        max_w10m_12 = np.max(the_common_data.loc[index, the_before_12_hour_w10m_obs_list])
        max_w10m_24 = np.max(the_common_data.loc[index, the_before_24_hour_w10m_obs_list])
        max_w10m_36 = np.max(the_common_data.loc[index, the_before_36_hour_w10m_obs_list])
        max_w10m_72 = np.max(the_common_data.loc[index, the_before_72_hour_w10m_obs_list])
        max_w10m_108 = np.max(the_common_data.loc[index, the_before_108_hour_w10m_obs_list])

        t2m_13 = mean_t2m_24 / mean_t2m_12
        t2m_35 = mean_t2m_36 / mean_t2m_24
        t2m_57 = mean_t2m_72 / mean_t2m_36
        t2m_79 = mean_t2m_108 / mean_t2m_72

        rh2m_13 = mean_rh2m_24 / mean_rh2m_12
        rh2m_35 = mean_rh2m_36 / mean_rh2m_24
        rh2m_57 = mean_rh2m_72 / mean_rh2m_36
        rh2m_79 = mean_rh2m_108 / mean_rh2m_72

        w10m_13 = mean_w10m_24 / mean_w10m_12
        w10m_35 = mean_w10m_36 / mean_w10m_24
        w10m_57 = mean_w10m_72 / mean_w10m_36
        w10m_79 = mean_w10m_108 / mean_w10m_72



        list_stat_number = [mean_t2m_12, mean_t2m_24, mean_t2m_36, min_t2m_12, min_t2m_24, min_t2m_36, max_t2m_12,
                            max_t2m_24
            , max_t2m_36, mean_rh2m_12, mean_rh2m_24, mean_rh2m_36, min_rh2m_12, min_rh2m_24, min_rh2m_36, max_rh2m_12,
                            max_rh2m_24
            , max_rh2m_36, mean_w10m_12, mean_w10m_24, mean_w10m_36, min_w10m_12, min_w10m_24, min_w10m_36, max_w10m_12,
                            max_w10m_24
            , max_w10m_36, t2m_13, t2m_35, rh2m_13, rh2m_35, w10m_13, w10m_35,
                            ]

        # 错了，少写个rh2m_57，算了不管了，有时间再改过来
        list_stat_number.extend(
            [mean_t2m_72, mean_t2m_108, min_t2m_72, min_t2m_108, max_t2m_72, max_t2m_108, mean_rh2m_72, mean_rh2m_108,
             min_rh2m_72, min_rh2m_108, max_rh2m_72, max_rh2m_108, mean_w10m_72, mean_w10m_108, min_w10m_72,
             min_w10m_108, max_w10m_72, max_w10m_108, t2m_57, t2m_79, rh2m_57, rh2m_79, w10m_57, w10m_79])



        for clo_num in range(0, 33):
            # print('长度检验 station:',len(list(the_station_forma.loc[index])))
            # print('长度检验 common:',len(list(the_common_data.loc[index])))
            # 进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_t2m = np.hstack((the_station_forma[index], list(the_common_data.loc[index]),list_stat_number, [clo_num + 1],
                                   list(the_flag_data.loc[index, [t2m_obs_clo[clo_num]]]),))
            t2m_all_data.append(every_t2m)

        print('-----  预测rh2m_obs模型需要的数据  -----')

        for clo_num in range(0, 33):
            # 进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_rh2m = np.hstack((the_station_forma[index], list(the_common_data.loc[index]),list_stat_number, [clo_num + 1],
                                    list(the_flag_data.loc[index, [rh2m_obs_clo[clo_num]]]),))
            rh2m_all_data.append(every_rh2m)

        print('-----  预测w10m_obs模型需要的数据  -----')

        for clo_num in range(0, 33):
            # 进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_obs = np.hstack((the_station_forma[index], list(the_common_data.loc[index]),list_stat_number, [clo_num + 1],
                                   list(the_flag_data.loc[index, [w10m_obs_clo[clo_num]]]),))
            w10m_all_data.append(every_obs)

    print('---------   对三种数据记性保存   ----------')
    f = h5py.File('./for_submit_do_data/t2m_all_data.h5', 'w')
    f['t2m_all_data'] = t2m_all_data
    f = h5py.File('./for_submit_do_data/rh2m_all_data.h5', 'w')
    f['rh2m_all_data'] = rh2m_all_data
    f = h5py.File('./for_submit_do_data/w10m_all_data.h5', 'w')
    f['w10m_all_data'] = w10m_all_data



def make_feature():
    print('--------   直接借鉴之前的方式，这里直接生成时序特征和统计特征   --------')

    print('------   生成日期点(之前在do_slice放置导致meomeryError,所以只能这样办了)   ------')
    with open('./for_submit_do_data/the_time.pkl', 'rb') as f:
        the_date_pkl = pickle.load(f)
    #构造序列中每个点的日期
    the_all_date=[]
    for i in the_date_pkl:
        for j in range(0,33):
            the_all_date.append(i)

    print('日期数量是：', len(the_all_date))
    print('------------    读取数据   ------------')
    #读取列名
    all_col_names = list(pd.read_csv('../feature_data/feature_col_name.csv')['0'])



    f = h5py.File('./for_submit_do_data/t2m_all_data.h5', 'r')
    t2m_data = f['t2m_all_data'].value
    f = h5py.File('./for_submit_do_data/rh2m_all_data.h5', 'r')
    rh2m_data = f['rh2m_all_data'].value
    f = h5py.File('./for_submit_do_data/w10m_all_data.h5', 'r')
    w10m_data = f['w10m_all_data'].value
    t2m_data_fm = pd.DataFrame(t2m_data,columns=all_col_names)
    rh2m_data_fm = pd.DataFrame(rh2m_data,columns=all_col_names)
    w10m_data_fm = pd.DataFrame(w10m_data,columns=all_col_names)



    # 选择性的dsa
    print('------------    删除最前面60个小时（也就是61到120小时）的历史特征（因为内存爆了）   ------------')
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    iiiii = 0
    the_columns_name = []
    for i in range(61, 121):
        for j in guance_columns:
            print(':::::', iiiii)
            the_columns_name.append(('guance_before_hour_' + str(i) + '_' + j))
            iiiii = iiiii + 1
    the_use_colu = [i for i in all_col_names if i not in the_columns_name]


    print('标示位置1：',len(the_use_colu))

    t2m_data_fm = t2m_data_fm[the_use_colu]
    rh2m_data_fm = rh2m_data_fm[the_use_colu]
    w10m_data_fm = w10m_data_fm[the_use_colu]




    # 读取假期特性
    holiday_data=pd.read_csv('../data/holiday_china_all.csv')
    holiday_data['date'] = holiday_data['date'].apply(str)
    print('------------   开始提取特征    ------------   ')
    print('----  首先根据日期构造时间特征（只是用日期就ok了）  ----')
    #是否工作日，是否星期天，是否工作日的最后一天，是否星期天的最后一天
    is_weekday=[]
    is_workday=[]
    is_holiday=[]
    is_weekday_after_24=[]
    is_workday_after_24=[]
    is_holiday_after_24=[]
    the_hour=[]
    the_week=[]
    for i in the_all_date:
        # 工作日对应结果为 0, 休息日对应结果为 1, 节假日对应的结果为 2   2018-08-30 19:00:00

        the_date = datetime.datetime(int(i[0:4]), int(i[5:7]), int(i[8:10]), int(i[11:13]))
        print(the_date)
        print('周几：', the_date.isoweekday())
        print('哪一天：', the_date.day)

        the_hour.append(int(i[11:13]))
        the_week.append(int(the_date.isoweekday()))
        pan_holiday = holiday_data[
            holiday_data['date'] == (str(the_date)[0:4] + str(the_date)[5:7] + str(the_date)[8:10])]

        date_flag = pan_holiday['holiday'].values[0]
        if date_flag == 0:
            # 为工作日
            is_workday.append(1)
            is_weekday.append(0)
            is_holiday.append(0)
        elif date_flag == 1:
            # 为休息日
            is_workday.append(0)
            is_weekday.append(1)
            is_holiday.append(0)
        elif date_flag == 2:
            # 为休息日
            is_workday.append(0)
            is_weekday.append(0)
            is_holiday.append(1)
        print('----  对于后24个小时的情况进行统计  ----')

        i = pd.to_datetime(i)
        the_date = i + datetime.timedelta(hours=24)

        pan_holiday = holiday_data[
            holiday_data['date'] == (str(the_date)[0:4] + str(the_date)[5:7] + str(the_date)[8:10])]

        date_flag = pan_holiday['holiday'].values[0]
        if date_flag == 0:
            # 为工作日
            is_workday_after_24.append(1)
            is_weekday_after_24.append(0)
            is_holiday_after_24.append(0)
        elif date_flag == 1:
            # 为休息日
            is_workday_after_24.append(0)
            is_weekday_after_24.append(1)
            is_holiday_after_24.append(0)
        elif date_flag == 2:
            # 为休息日
            is_workday_after_24.append(0)
            is_weekday_after_24.append(0)
            is_holiday_after_24.append(1)



    print('-----    将时间属性特征填充过去   -----')
    print('不匹配的话，要提充的长度：', is_weekday)
    print('大的表格的长度：', t2m_data_fm.iloc[:, 0].size)
    t2m_data_fm['is_weekday'] = is_weekday
    t2m_data_fm['is_workday'] = is_workday
    t2m_data_fm['is_holiday'] = is_holiday
    t2m_data_fm['is_workday_after_24'] = is_workday_after_24
    t2m_data_fm['is_weekday_after_24'] = is_weekday_after_24
    t2m_data_fm['is_holiday_after_24'] = is_holiday_after_24
    t2m_data_fm['start_hour'] = the_hour
    t2m_data_fm['the_week'] = the_week

    rh2m_data_fm['is_weekday'] = is_weekday
    rh2m_data_fm['is_workday'] = is_workday
    rh2m_data_fm['is_holiday'] = is_holiday
    rh2m_data_fm['is_workday_after_24'] = is_workday_after_24
    rh2m_data_fm['is_weekday_after_24'] = is_weekday_after_24
    rh2m_data_fm['is_holiday_after_24'] = is_holiday_after_24
    rh2m_data_fm['start_hour'] = the_hour
    rh2m_data_fm['the_week'] = the_week

    w10m_data_fm['is_weekday'] = is_weekday
    w10m_data_fm['is_workday'] = is_workday
    w10m_data_fm['is_holiday'] = is_holiday
    w10m_data_fm['is_workday_after_24'] = is_workday_after_24
    w10m_data_fm['is_weekday_after_24'] = is_weekday_after_24
    w10m_data_fm['is_holiday_after_24'] = is_holiday_after_24
    w10m_data_fm['start_hour'] = the_hour
    w10m_data_fm['the_week'] = the_week

    print('------   将经过特征工程处理过的大Dataframe进行保存   ------')
    print('------   将经过特征工程处理过的大Dataframe进行保存   ------')

    print('标示位置2：',len(t2m_data_fm.columns))

    print('小时信息：',t2m_data_fm['the_hour'])
    print('mean_t2m_12信息：',t2m_data_fm['mean_t2m_12'])
    print('res_value信息：',t2m_data_fm['res_value'])


    t2m_data_fm.to_csv('./for_submit_do_data/the_final_t2m_gen_fea_data.csv')
    rh2m_data_fm.to_csv('./for_submit_do_data/the_final_rh2m_gen_fea_data.csv')
    w10m_data_fm.to_csv('./for_submit_do_data/the_final_w10m_gen_fea_data.csv')



def do_predict_t2m():
    '''
    在预测结果中生成了
    '''
    print('--------------   读取模型和训练好的特征数据启动预测   ------------')
    print('--------  产生预测结果  --------')
    data=pd.read_csv('./for_submit_do_data/the_final_t2m_gen_fea_data.csv',index_col=0)


    print('标示位置3：',len(data.columns))
    data.pop('res_value')
    X = data
    model = pickle.load(open('../model/save_model/lgb_t2m.model', "rb"))
    the_pred_t2m=model.predict(X)

    print('预测值为：',the_pred_t2m)

    data['t2m']=the_pred_t2m

    print('--------  生成提交文件(先拼接预测结果)  --------')
    #生成对应的站点名称
    AnEn_data_list_part1=[]
    for i in ['90001','90002','90003','90004','90005','90006','90007','90008','90009','90010']:
        for j in range(4,37):
            if j<10:
                AnEn_data_list_part1.append(i+'0'+str(j))
            else:
                AnEn_data_list_part1.append(i + str(j))
    data['AnEn_data']=AnEn_data_list_part1
    the_sub_part1=data[['AnEn_data','t2m']]
    print('--------  拼接已有的前三个小时的真实值  --------')
    #读取gc文件从中筛选出满足要求的对应三个小时的指标值，用于拼接到提交文件中
    gc_data=pd.read_csv('./for_submit_do_data/do_fill_na_gc.csv',index_col=0)
    gc_data['time'] = pd.to_datetime(gc_data['time'])
    the_start_slide_date = pd.to_datetime('2018/10/15 03:00:00')
    end_time=the_start_slide_date+datetime.timedelta(hours=3)

    the_big_t2m_list=[]
    for station in [90001,90002,90003,90004,90005,90006,90007,90008,90009,90010]:
        the_first_range_data=gc_data[(gc_data['the_station']==station)&(gc_data['time']>=the_start_slide_date)&(gc_data['time']<=end_time)]
        the_big_t2m_list.extend(list(the_first_range_data['t2m_obs']))
    print(len(the_big_t2m_list))

    AnEn_data_list_part2= []
    for i in ['90001','90002','90003','90004','90005','90006','90007','90008','90009','90010']:
        for j in range(0,4):
            AnEn_data_list_part2.append(i+'0'+str(j))
    the_sub_part2=pd.DataFrame(columns=['AnEn_data','t2m'])
    the_sub_part2['AnEn_data']=AnEn_data_list_part2
    the_sub_part2['t2m']=the_big_t2m_list
    print('--------  进行两部分提交文件的拼接  --------')
    the_submit_file=pd.concat([the_sub_part1,the_sub_part2],axis=0)

    print('--------  对站点序号进行排序  --------')
    the_submit_file['AnEn_data']=the_submit_file['AnEn_data'].apply(pd.to_numeric)
    the_submit_file=the_submit_file.sort_values(by=["AnEn_data"])
    the_submit_file['AnEn_data']=[str(i)[0:5]+'_'+str(i)[5:7]  for i in the_submit_file['AnEn_data']]
    the_submit_file.to_csv('./submit/submit_t2m_'+ str(datetime.datetime.now())[0:11]+'.csv')

def do_predict_rh2m():
    '''
    在预测结果中生成了
    '''
    print('--------------   读取模型和训练好的特征数据启动预测   ------------')
    print('--------  产生预测结果  --------')
    data=pd.read_csv('./for_submit_do_data/the_final_rh2m_gen_fea_data.csv',index_col=0)
    data.pop('res_value')
    X = data
    model = pickle.load(open('../model/save_model/lgb_rh2m.model', "rb"))
    the_pred_rh2m = model.predict(X)
    data['rh2m'] = the_pred_rh2m

    print('--------  生成提交文件(先拼接预测结果)  --------')
    # 生成对应的站点名称
    AnEn_data_list_part1 = []
    for i in ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009', '90010']:
        for j in range(4, 37):
            if j < 10:
                AnEn_data_list_part1.append(i + '0' + str(j))
            else:
                AnEn_data_list_part1.append(i + str(j))
    data['AnEn_data'] = AnEn_data_list_part1
    the_sub_part1 = data[['AnEn_data', 'rh2m']]
    print('--------  拼接已有的前三个小时的真实值  --------')
    # 读取gc文件从中筛选出满足要求的对应三个小时的指标值，用于拼接到提交文件中
    gc_data = pd.read_csv('./for_submit_do_data/do_fill_na_gc.csv', index_col=0)
    gc_data['time'] = pd.to_datetime(gc_data['time'])
    the_start_slide_date = pd.to_datetime('2018/10/15 03:00:00')
    end_time = the_start_slide_date + datetime.timedelta(hours=3)

    the_big_t2m_list = []
    for station in [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009, 90010]:
        the_first_range_data = gc_data[
            (gc_data['the_station'] == station) & (gc_data['time'] >= the_start_slide_date) & (
                        gc_data['time'] <= end_time)]
        the_big_t2m_list.extend(list(the_first_range_data['rh2m_obs']))
    print(len(the_big_t2m_list))

    AnEn_data_list_part2 = []
    for i in ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009', '90010']:
        for j in range(0, 4):
            AnEn_data_list_part2.append(i + '0' + str(j))
    the_sub_part2 = pd.DataFrame(columns=['AnEn_data', 'rh2m'])
    the_sub_part2['AnEn_data'] = AnEn_data_list_part2
    the_sub_part2['rh2m'] = the_big_t2m_list
    print('--------  进行两部分提交文件的拼接  --------')
    the_submit_file = pd.concat([the_sub_part1, the_sub_part2], axis=0)

    print('--------  对站点序号进行排序  --------')
    the_submit_file['AnEn_data'] = the_submit_file['AnEn_data'].apply(pd.to_numeric)
    the_submit_file = the_submit_file.sort_values(by=["AnEn_data"])
    the_submit_file['AnEn_data'] = [str(i)[0:5] + '_' + str(i)[5:7] for i in the_submit_file['AnEn_data']]
    the_submit_file.to_csv('./submit/submit_rh2m_' + str(datetime.datetime.now())[0:11] + '.csv')

def do_predict_w10m():
    '''
    在预测结果中生成了
    '''
    print('--------------   读取模型和训练好的特征数据启动预测   ------------')
    print('--------  产生预测结果  --------')
    data=pd.read_csv('./for_submit_do_data/the_final_w10m_gen_fea_data.csv',index_col=0)
    data.pop('res_value')
    X = data
    model = pickle.load(open('../model/save_model/lgb_w10m.model', "rb"))
    the_pred_w10m = model.predict(X)
    data['w10m'] = the_pred_w10m

    print('--------  生成提交文件(先拼接预测结果)  --------')
    # 生成对应的站点名称
    AnEn_data_list_part1 = []
    for i in ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009', '90010']:
        for j in range(4, 37):
            if j < 10:
                AnEn_data_list_part1.append(i + '0' + str(j))
            else:
                AnEn_data_list_part1.append(i + str(j))
    data['AnEn_data'] = AnEn_data_list_part1
    the_sub_part1 = data[['AnEn_data', 'w10m']]
    print('--------  拼接已有的前三个小时的真实值  --------')
    # 读取gc文件从中筛选出满足要求的对应三个小时的指标值，用于拼接到提交文件中
    gc_data = pd.read_csv('./for_submit_do_data/do_fill_na_gc.csv', index_col=0)
    gc_data['time'] = pd.to_datetime(gc_data['time'])
    the_start_slide_date = pd.to_datetime('2018/10/15 03:00:00')
    end_time = the_start_slide_date + datetime.timedelta(hours=3)

    the_big_t2m_list = []
    for station in [90001, 90002, 90003, 90004, 90005, 90006, 90007, 90008, 90009, 90010]:
        the_first_range_data = gc_data[
            (gc_data['the_station'] == station) & (gc_data['time'] >= the_start_slide_date) & (
                        gc_data['time'] <= end_time)]
        the_big_t2m_list.extend(list(the_first_range_data['w10m_obs']))
    print(len(the_big_t2m_list))

    AnEn_data_list_part2 = []
    for i in ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009', '90010']:
        for j in range(0, 4):
            AnEn_data_list_part2.append(i + '0' + str(j))
    the_sub_part2 = pd.DataFrame(columns=['AnEn_data', 'w10m'])
    the_sub_part2['AnEn_data'] = AnEn_data_list_part2
    the_sub_part2['w10m'] = the_big_t2m_list
    print('--------  进行两部分提交文件的拼接  --------')
    the_submit_file = pd.concat([the_sub_part1, the_sub_part2], axis=0)

    print('--------  对站点序号进行排序  --------')
    the_submit_file['AnEn_data'] = the_submit_file['AnEn_data'].apply(pd.to_numeric)
    the_submit_file = the_submit_file.sort_values(by=["AnEn_data"])
    the_submit_file['AnEn_data'] = [str(i)[0:5] + '_' + str(i)[5:7] for i in the_submit_file['AnEn_data']]
    the_submit_file.to_csv('./submit/submit_w10m_' + str(datetime.datetime.now())[0:11] + '.csv')


def concat_submit():
    t2m=pd.read_csv('./submit/submit_t2m_' + str(datetime.datetime.now())[0:11] + '.csv',index_col=0)
    rh2m=pd.read_csv('./submit/submit_rh2m_' + str(datetime.datetime.now())[0:11] + '.csv',index_col=0)
    w10m=pd.read_csv('./submit/submit_w10m_' + str(datetime.datetime.now())[0:11] + '.csv',index_col=0)
    the_big=pd.merge(t2m,rh2m,on='AnEn_data')
    the_big=pd.merge(the_big,w10m,on='AnEn_data')
    print(the_big.head())
    the_big.to_csv('./submit/submit_all_ceshi_10_30_1.csv',index=False)
def do_pipeline():
    '''
        构建自动化提交的过程，包含以下几步：
               *从nc文件中抽取出要预测点之前的观测值 和 睿图数据（其实只需要这个时间之前的两周就肯定够了）
               *对拿出来的数据做预处理
               *进行时序调整，获取那个时间点的滑动，会滑动出（34*10）行数据，其中前三个小时的3*10条数据是已经给出的.....
               *对着340行数据抽取构建出时间属性、时序特征
               *模型预测
               *生成预测结果进行填充。
        真是没得说，妈的，3点的数据是没给的，只是给了4、5、6三个小时的数据，搞得划取特征时候，滑错了，真是闹着玩，每天凌晨3点数据都是没有的。
        算了，发现好多都是凌晨3点数据都是没有的，算了，还是老方法建模吧，创建出3个模型来，预测未来34个小时的数据，前三个小时的加进去。
    '''
    print('-------   对最新提供的nc文件进行解析（这里以第二次双周赛的数据来做实验）   -------')
    # #传递参数，
    # nc_to_csv_ruitu()
    # nc_to_csv_gc()
    # # # # print('-------   数据预处理   -------')
    # do_data_process()
    # # # # # # print('-------   构造时序数据   -------')
    # gen_time_window_data()
    # gen_time_window_data_gen_flag()
    # # # # # # print('-------   抽取造特征   -------')
    # make_feature()
    # # # print('-------   模型预测   -------')
    # # # #这里暂时只是举单个模型为例子
    # do_predict_t2m()
    # do_predict_rh2m()
    # do_predict_w10m()

    concat_submit()
    print('-------   构造提交文件   -------')


if __name__=='__main__':


    do_pipeline()

