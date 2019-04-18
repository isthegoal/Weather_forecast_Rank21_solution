# -*- coding:utf-8 -*-
'''
整体思路是从nc文件中提取每个时刻点的睿图信息和观测信息。   保存到csv文件中去....
'''
import netCDF4 as nc
import sys
import glob
import csv
import numpy as np
import pandas as pd
import time
import datetime
'''
初始构建：
    睿图数据包括： 天日期（基础日期加日期推断），小时，站点，睿图元素数据                   可以看出是以一个时间点为单位的
    观测表数据包括：天日期（基础日期加日期推断），小时，站点，观测真实元素数据
'''
def nc_to_csv_ruitu():
    print('1111')
    the_test_data = nc.Dataset('./data/ai_challenger_wf2018_testa1_20180829-20180924.nc')
    the_train_data = nc.Dataset('./data/ai_challenger_wf2018_trainingset_20150301-20180531.nc')
    the_val_data = nc.Dataset('./data/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc')
    the_testB_data = nc.Dataset('./data/ai_challenger_weather_testingsetB_20180829-20181015.nc')

    the_test_final_data = nc.Dataset('./data/ai_challenger_wf2018_testb1_20180829-20181028.nc')


    ruitu_columns=['psfc_M','t2m_M','q2m_M','rh2m_M','w10m_M','d10m_M','u10m_M','v10m_M','SWD_M','GLW_M','HFX_M','LH_M','RAIN_M','PBLH_M','TC975_M','TC925_M','TC850_M','TC700_M','TC500_M','wspd975_M','wspd925_M','wspd850_M','wspd700_M','wspd500_M','Q975_M','Q925_M','Q850_M','Q700_M','Q500_M']
    guance_columns=['psur_obs','t2m_obs','q2m_obs','rh2m_obs','w10m_obs','d10m_obs','u10m_obs','v10m_obs','RAIN_obs']
    print(the_train_data['psur_obs'])
    print('---------------------------  提取出睿图信息  -------------------------------')

    the_big_ruitu_table=[]

    the_all_time=[]


    print('-- 对有61天的测试集3[最终数据集]进行处理(61, 37, 10) --')
    for i in range(0, 61):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_test_final_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            # print(the_real_data)

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in ruitu_columns:
                    the_hang.append(np.array(the_test_final_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_ruitu_table.append(the_hang)


    print('-- 对有48天的测试集2进行处理(48, 37, 10) --')
    # 对于the_testB_data
    for i in range(0, 48):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_testB_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            # print(the_real_data)

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in ruitu_columns:
                    the_hang.append(np.array(the_testB_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_ruitu_table.append(the_hang)

            # print('要加入的一行是',the_big_ruitu_table)

    print('-- 对有27天的测试集1进行处理(27, 37, 10) --')
    # 对于the_test_data
    for i in range(0, 27):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_test_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in ruitu_columns:
                    the_hang.append(np.array(the_test_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_ruitu_table.append(the_hang)

            # print('要加入的一行是',the_big_ruitu_table)
    # print('-- 对有1188天的训练集进行处理(1188, 37, 10) --')
    # # # 数据中（天数，时刻，指标值）
    # for i in range(0, 1188):
    #     # 对于每一天，获取对应37个时刻中的每一个时刻
    #     for j in range(0,37):
    #         #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
    #         #第一步加入时间点
    #         the_ori_date=the_train_data['date'][i]
    #         the_formated_date=pd.to_datetime(str(the_ori_date)[:-2],format='%Y%m%d%H')
    #         the_real_data=the_formated_date+datetime.timedelta(hours=j)
    #
    #         if the_real_data not in the_all_time:
    #             the_all_time.append(the_real_data)
    #         else:
    #             continue
    #         #print(the_real_data)
    #         #第二步加入站点信息
    #         the_station_name=['90001', '90002', '90003', '90004' ,'90005' ,'90006' ,'90007' ,'90008' ,'90009', '90010']
    #         for k in range(0,10):
    #             the_hang = []#不断对于每个站点都这样进行加入
    #             the_hang.append(the_real_data)
    #             the_hang.append(the_station_name[k])
    #             #第三步加入每个属性的信息
    #             for t in  ruitu_columns:
    #                 the_hang.append(np.array(the_train_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据
    #
    #             the_big_ruitu_table.append(the_hang)

            #print('要加入的一行是',the_big_ruitu_table)
    print('-- 对有89天的验证集进行处理(89, 37, 10) --')
    #对于the_val_data中的
    for i in range(0, 89):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0,37):
            #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            #第一步加入时间点
            the_ori_date=the_val_data['date'][i]
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
                    the_hang.append(np.array(the_val_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据
                the_big_ruitu_table.append(the_hang)




    the_ruitu_df=pd.DataFrame(the_big_ruitu_table,columns=['time','the_station','psfc_M','t2m_M','q2m_M','rh2m_M','w10m_M','d10m_M','u10m_M','v10m_M','SWD_M','GLW_M','HFX_M','LH_M','RAIN_M','PBLH_M','TC975_M','TC925_M','TC850_M','TC700_M','TC500_M','wspd975_M','wspd925_M','wspd850_M','wspd700_M','wspd500_M','Q975_M','Q925_M','Q850_M','Q700_M','Q500_M'])
    the_ruitu_df.to_csv('./data/ruitu_all_time_data.csv')

    pass


def nc_to_csv_gc():
    print('2222')
    the_test_data = nc.Dataset('./data/ai_challenger_wf2018_testa1_20180829-20180924.nc')
    the_train_data = nc.Dataset('./data/ai_challenger_wf2018_trainingset_20150301-20180531.nc')
    the_val_data = nc.Dataset('./data/ai_challenger_wf2018_validation_20180601-20180828_20180905.nc')
    the_testB_data = nc.Dataset('./data/ai_challenger_weather_testingsetB_20180829-20181015.nc')

    the_test_final_data = nc.Dataset('./data/ai_challenger_wf2018_testb1_20180829-20181028.nc')

    ruitu_columns=['psfc_M','t2m_M','q2m_M','rh2m_M','w10m_M','d10m_M','u10m_M','v10m_M','SWD_M','GLW_M','HFX_M','LH_M','RAIN_M','PBLH_M','TC975_M','TC925_M','TC850_M','TC700_M','TC500_M','wspd975_M','wspd925_M','wspd850_M','wspd700_M','wspd500_M','Q975_M','Q925_M','Q850_M','Q700_M','Q500_M']
    guance_columns=['psur_obs','t2m_obs','q2m_obs','rh2m_obs','w10m_obs','d10m_obs','u10m_obs','v10m_obs','RAIN_obs']
    print(the_train_data['psur_obs'])
    print('---------------------------  提取出观测信息  -------------------------------')

    the_big_gc_table=[]
    the_all_time=[]

    print('-- 对有61天的测试集3[最终数据集]进行处理(61, 37, 10) --')
    for i in range(0, 61):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_test_final_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            # print(the_real_data)

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in guance_columns:
                    the_hang.append(np.array(the_test_final_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_gc_table.append(the_hang)

    print('-- 对有48天的测试集2进行处理(48, 37, 10) --')
    # 对于the_testB_data
    for i in range(0, 48):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_testB_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            # print(the_real_data)

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in guance_columns:
                    the_hang.append(np.array(the_testB_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_gc_table.append(the_hang)

    print('-- 对有27天的测试集1进行处理(27, 37, 10) --')
    # 对于the_test_data
    for i in range(0, 27):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0, 37):
            # 一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            # 第一步加入时间点
            the_ori_date = the_test_data['date'][i]
            the_formated_date = pd.to_datetime(str(the_ori_date)[:-2], format='%Y%m%d%H')
            the_real_data = the_formated_date + datetime.timedelta(hours=j)
            if the_real_data not in the_all_time:
                the_all_time.append(the_real_data)
            else:
                continue
            # print(the_real_data)

            # 第二步加入站点信息
            the_station_name = ['90001', '90002', '90003', '90004', '90005', '90006', '90007', '90008', '90009',
                                '90010']
            for k in range(0, 10):
                the_hang = []  # 不断对于每个站点都这样进行加入
                the_hang.append(the_real_data)
                the_hang.append(the_station_name[k])
                # 第三步加入每个属性的信息
                for t in guance_columns:
                    the_hang.append(np.array(the_test_data[t])[i][j][k])  # 针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_gc_table.append(the_hang)

            # print('要加入的一行是',the_big_gc_table)
    print('-- 对有1188天的训练集进行处理(1188, 37, 10) --')
    # 数据中（天数，时刻，指标值）
    # for i in range(0, 1188):
    #     # 对于每一天，获取对应37个时刻中的每一个时刻
    #     for j in range(0,37):
    #         #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
    #         #第一步加入时间点
    #         the_ori_date=the_train_data['date'][i]
    #         the_formated_date=pd.to_datetime(str(the_ori_date)[:-2],format='%Y%m%d%H')
    #         the_real_data=the_formated_date+datetime.timedelta(hours=j)
    #         if the_real_data not in the_all_time:
    #             the_all_time.append(the_real_data)
    #         else:
    #             continue
    #         #print(the_real_data)
    #
    #         #第二步加入站点信息
    #         the_station_name=['90001', '90002', '90003', '90004' ,'90005' ,'90006' ,'90007' ,'90008' ,'90009', '90010']
    #         for k in range(0,10):
    #             the_hang = []#不断对于每个站点都这样进行加入
    #             the_hang.append(the_real_data)
    #             the_hang.append(the_station_name[k])
    #             #第三步加入每个属性的信息
    #             for t in  guance_columns:
    #                 the_hang.append(np.array(the_train_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据
    #
    #             the_big_gc_table.append(the_hang)

            #print('要加入的一行是',the_big_gc_table)
    print('-- 对有89天的验证集进行处理(89, 37, 10) --')
    #对于the_val_data中的
    for i in range(0, 89):
        # 对于每一天，获取对应37个时刻中的每一个时刻
        for j in range(0,37):
            #一个个加进去，分别加入具体时间点    站点信息    这一小时下的所有睿图数据
            #第一步加入时间点
            the_ori_date=the_val_data['date'][i]
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
                for t in  guance_columns:
                    the_hang.append(np.array(the_val_data[t])[i][j][k])   #针对哪个属性的  哪天哪个小时哪个站点的数据

                the_big_gc_table.append(the_hang)

            #print('要加入的一行是',the_big_gc_table)




            #print('要加入的一行是',the_big_gc_table)

    the_gc_df=pd.DataFrame(the_big_gc_table,columns=['time','the_station','psur_obs','t2m_obs','q2m_obs','rh2m_obs','w10m_obs','d10m_obs','u10m_obs','v10m_obs','RAIN_obs'])
    the_gc_df.to_csv('./data/gc_all_time_data.csv')

    pass

if __name__=='__main__':
    print('---------------------------  提取出睿图信息  ---------------------------')
    nc_to_csv_ruitu()
    print('---------------------------  提取出观测气象信息  ---------------------------')
    nc_to_csv_gc()
    print('ok')


