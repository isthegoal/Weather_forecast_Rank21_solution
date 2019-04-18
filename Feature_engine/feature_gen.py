# coding=utf-8
import pandas as pd
import numpy as np
import h5py
import pickle
import time
from sklearn import preprocessing
from datetime import datetime
import datetime
from model.train_lightGBM import train_lgb_model_t2m,train_lgb_model_rh2m,train_lgb_model_w10m
import urllib.request
import json


server_url = "http://api.goseek.cn/Tools/holiday?date="
'''
构建特征思路：
      *判断开始的这个时间点是否是周末、工作日第一天、工作日最后一天，这个时间点25小时后的时间点是够周末、工作日第一天、工作日最后一天
      *用前5天数据计算9种观测指标的最大、最小、平均值，前一天的最大值最小平均值、前10个小时的最大最小平均值
      
      
获取节假日信息，可以使用http://api.goseek.cn/Tools/holiday?date=20180922，其中
     工作日对应结果为 0, 休息日对应结果为 1, 节假日对应的结果为 2 
'''
def gen_col_name():
    # print('6666')
    # print('------------   提取初始构建的h5文件   ------------')
    # f = h5py.File('./feature_data/t2m_all_data.h5', 'r')
    # t2m_data = f['t2m_all_data'].value
    # f = h5py.File('./feature_data/rh2m_all_data.h5', 'r')
    # rh2m_data = f['rh2m_all_data'].value
    # f = h5py.File('./feature_data/w10m_all_data.h5', 'r')
    # w10m_data = f['w10m_all_data'].value
    #
    # weidu_size=t2m_data.shape[1]
    print('------------   对应生成列名   ------------')
    ruitu_columns = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M',
                     'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M',
                     'wspd975_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
                     'Q700_M', 'Q500_M']
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    the_columns_name=[]
    #第一部分
    #the_columns_name.append('time')
    the_columns_name.append('station')
    #第二部分  40个观测信息（横向的小时里的合在一起，反正无时间点缺失）
    for i in range(0,120):
        for j in guance_columns:
            the_columns_name.append('guance_before_hour_'+str(120-i)+'_'+j)
    print('长度', len(the_columns_name))
    # 第三部分  37个未来睿图信息
    for i in range(0,37):
        for j in ruitu_columns:
            the_columns_name.append('ruitu_future_hour_'+str(i)+'_'+j)

    print('长度', len(the_columns_name))
    # 第四部分  3个未来观测信息
    for i in range(0,4):
        for j in guance_columns:
            the_columns_name.append('guance_future_hour_' + str(i) + '_' + j)

    print('长度', len(the_columns_name))
    # 第五部分  统计特征

    the_columns_name.extend(['mean_t2m_12',  'mean_t2m_24',  'mean_t2m_36',  'min_t2m_12',  'min_t2m_24',  'min_t2m_36',  'max_t2m_12','max_t2m_24', 'max_t2m_36',
                             'mean_rh2m_12', 'mean_rh2m_24', 'mean_rh2m_36', 'min_rh2m_12', 'min_rh2m_24', 'min_rh2m_36', 'max_rh2m_12','max_rh2m_24', 'max_rh2m_36',
                             'mean_w10m_12', 'mean_w10m_24', 'mean_w10m_36', 'min_w10m_12', 'min_w10m_24', 'min_w10m_36', 'max_w10m_12','max_w10m_24', 'max_w10m_36',
                             't2m_13','t2m_35','rh2m_13','rh2m_35','w10m_13','w10m_35'])
    the_columns_name.extend(
        ['mean_t2m_72', 'mean_t2m_108', 'min_t2m_72', 'min_t2m_108', 'max_t2m_72', 'max_t2m_108', 'mean_rh2m_72', 'mean_rh2m_108', 'min_rh2m_72', 'min_rh2m_108', 'max_rh2m_72', 'max_rh2m_108', 'mean_w10m_72', 'mean_w10m_108', 'min_w10m_72', 'min_w10m_108', 'max_w10m_72', 'max_w10m_108', 't2m_57', 't2m_79', 'rh2m_57', 'rh2m_79', 'w10m_57', 'w10m_79'])
    print('长度', len(the_columns_name))
    # 第六部分  预测小时序号   实际指标值
    the_columns_name.append('the_hour')
    the_columns_name.append('res_value')

    print('长度',len(the_columns_name))#这里长度是1473，说明要传递过来的数据也应该是 1473
    print('------------   生成列名并保存，但是每个文件就有4个G，真是恐怖，还是之前的全放在h5文件中3个G的比较好(数值用h5存储)，对列名单独进行保存  ------------')
    feature_col_name=pd.DataFrame(the_columns_name)
    feature_col_name.to_csv('./feature_data/feature_col_name.csv')
    print(feature_col_name)
def make_time_series_fea():
    print('7777')
    print('------   生成日期点(之前在do_slice放置导致meomeryError,所以只能这样办了)   ------')
    with open('./feature_data/the_time.pkl', 'rb') as f:
        the_date_pkl = pickle.load(f)
    #构造序列中每个点的日期
    the_all_date=[]
    for i in the_date_pkl:
        for j in range(0,33):
            the_all_date.append(i)

    print('日期数量是：', len(the_all_date))
    print('------------    读取数据   ------------')
    #读取列名
    all_col_names = list(pd.read_csv('./feature_data/feature_col_name.csv')['0'])

    f = h5py.File('./feature_data/t2m_all_data.h5', 'r')
    t2m_data = f['t2m_all_data'].value
    f = h5py.File('./feature_data/rh2m_all_data.h5', 'r')
    rh2m_data = f['rh2m_all_data'].value
    f = h5py.File('./feature_data/w10m_all_data.h5', 'r')
    w10m_data = f['w10m_all_data'].value
    t2m_data_fm = pd.DataFrame(t2m_data,columns=all_col_names)
    rh2m_data_fm = pd.DataFrame(rh2m_data,columns=all_col_names)
    w10m_data_fm = pd.DataFrame(w10m_data,columns=all_col_names)

    #选择性的dsa
    print('------------    删除最前面60个小时（也就是61到120小时）的历史特征（因为内存爆了）   ------------')
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    iiiii=0
    the_columns_name=[]
    for i in range(61,121):
        for j in guance_columns:
            print(':::::',iiiii)
            the_columns_name.append(('guance_before_hour_'+str(i)+'_'+j))
            iiiii=iiiii+1
    the_use_colu=[i for i in all_col_names if i not in the_columns_name]
    t2m_data_fm=t2m_data_fm[the_use_colu]
    rh2m_data_fm=rh2m_data_fm[the_use_colu]
    w10m_data_fm=w10m_data_fm[the_use_colu]


    # 读取假期特性
    holiday_data=pd.read_csv('./data/holiday_china_all.csv')
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

    index=0
    for i in the_all_date:
        #工作日对应结果为 0, 休息日对应结果为 1, 节假日对应的结果为 2   2018-08-30 19:00:00

        the_date = datetime.datetime(int(i[0:4]),int(i[5:7]),int(i[8:10]),int(i[11:13]))
        print(the_date)
        print('周几：',the_date.isoweekday())
        print('哪一天：',the_date.day)

        the_hour.append(int(i[11:13]))
        the_week.append(int(the_date.isoweekday()))
        pan_holiday=holiday_data[holiday_data['date']==(str(the_date)[0:4] + str(the_date)[5:7] + str(the_date)[8:10])]

        date_flag=pan_holiday['holiday'].values[0]
        if date_flag==0:
            #为工作日
            is_workday.append(1)
            is_weekday.append(0)
            is_holiday.append(0)
        elif date_flag==1:
            #为休息日
            is_workday.append(0)
            is_weekday.append(1)
            is_holiday.append(0)
        elif date_flag==2:
            #为休息日
            is_workday.append(0)
            is_weekday.append(0)
            is_holiday.append(1)
        print('----  对于后24个小时的情况进行统计  ----')

        i = pd.to_datetime(i)
        the_date = i + datetime.timedelta(hours=24)

        pan_holiday=holiday_data[holiday_data['date']==(str(the_date)[0:4] + str(the_date)[5:7] + str(the_date)[8:10])]

        date_flag=pan_holiday['holiday'].values[0]
        if date_flag==0:
            #为工作日
            is_workday_after_24.append(1)
            is_weekday_after_24.append(0)
            is_holiday_after_24.append(0)
        elif date_flag==1:
            #为休息日
            is_workday_after_24.append(0)
            is_weekday_after_24.append(1)
            is_holiday_after_24.append(0)
        elif date_flag==2:
            #为休息日
            is_workday_after_24.append(0)
            is_weekday_after_24.append(0)
            is_holiday_after_24.append(1)

        index=index+1

    print('--------------       第一部分构造：对t2m_data_fm进行构造         ----------------')


    print('-----    将时间属性特征填充过去   -----')
    print('不匹配的话，要提充的长度：',is_weekday)
    print('大的表格的长度：',t2m_data_fm.iloc[:,0].size)
    t2m_data_fm['is_weekday']=is_weekday
    t2m_data_fm['is_workday']=is_workday
    t2m_data_fm['is_holiday']=is_holiday
    t2m_data_fm['is_workday_after_24']=is_workday_after_24
    t2m_data_fm['is_weekday_after_24']=is_weekday_after_24
    t2m_data_fm['is_holiday_after_24']=is_holiday_after_24
    t2m_data_fm['start_hour']=the_hour
    t2m_data_fm['the_week']=the_week

    rh2m_data_fm['is_weekday']=is_weekday
    rh2m_data_fm['is_workday']=is_workday
    rh2m_data_fm['is_holiday']=is_holiday
    rh2m_data_fm['is_workday_after_24']=is_workday_after_24
    rh2m_data_fm['is_weekday_after_24']=is_weekday_after_24
    rh2m_data_fm['is_holiday_after_24']=is_holiday_after_24
    rh2m_data_fm['start_hour']=the_hour
    rh2m_data_fm['the_week']=the_week

    w10m_data_fm['is_weekday']=is_weekday
    w10m_data_fm['is_workday']=is_workday
    w10m_data_fm['is_holiday']=is_holiday
    w10m_data_fm['is_workday_after_24']=is_workday_after_24
    w10m_data_fm['is_weekday_after_24']=is_weekday_after_24
    w10m_data_fm['is_holiday_after_24']=is_holiday_after_24
    w10m_data_fm['start_hour']=the_hour
    w10m_data_fm['the_week']=the_week

    print('------   将经过特征工程处理过的大Dataframe进行保存   ------')
    # print('大表列名是：',the_big_df.columns)
    # print(the_big_df.head()) #整好1505个   1472+6+27=1505



    print('---------   对三种数据记性保存 (h5文件只能保存数值，但是文件大小小得多，更加方便，不会出现error错误，pandas自带处理即可)  ----------')
    # f=pd.HDFStore('./feature_data/do_feature_eng_table/ver1_t2m_all_data.h5', 'w')
    # f['t2m_all_data'] = t2m_data_fm.values
    # f = pd.HDFStore('./feature_data/do_feature_eng_table/ver1_trh2m_all_data.h5', 'w')
    # f['rh2m_all_data'] = rh2m_data_fm.values
    # f = pd.HDFStore('./feature_data/do_feature_eng_table/ver1_tw10m_all_data.h5', 'w')
    # f['w10m_all_data'] = w10m_data_fm.values


    f = h5py.File('./feature_data/do_feature_eng_table/ver1_t2m_all_data.h5', 'w')
    f['t2m_all_data'] = t2m_data_fm.values
    f = h5py.File('./feature_data/do_feature_eng_table/ver1_trh2m_all_data.h5', 'w')
    f['rh2m_all_data'] = rh2m_data_fm.values
    f = h5py.File('./feature_data/do_feature_eng_table/ver1_tw10m_all_data.h5', 'w')
    f['w10m_all_data'] = w10m_data_fm.values
    the_columns_name=t2m_data_fm.columns
    feature_col_name=pd.DataFrame(the_columns_name)
    feature_col_name.to_csv('./feature_data/do_feature_eng_table/ver1_feature_col_name.csv')
    print(feature_col_name)
    # t2m_data_fm.to_csv('./feature_data/do_feature_eng_table/t2m_data_tabel.csv')
    # rh2m_data_fm.to_csv('./feature_data/do_feature_eng_table/rh2m_data_tabel.csv')
    # w10m_data_fm.to_csv('./feature_data/do_feature_eng_table/w10m_data_tabel.csv')

    '''
    版本1
    
    '''


def crawel_holiday():
    '''
    现在已经有20170101  到 20180531 的所有日期标识信息，我们继续往其中附加 20180601到20181105这些天的节假日信息
    '''
    start_data='2018-06-06 19:00:00'
    end_data = '2018-11-05 19:00:00'
    #end_data='2018-11-05 19:00:00'
    start_timestamp = pd.to_datetime(start_data)
    end_timestamp = pd.to_datetime(end_data)
    have_holiday=pd.read_csv('./data/holiday_bj.csv')
    have_holiday.pop('lunardate') #不要农历
    print('----    开始循环捕获日期    ----')
    while start_timestamp<=end_timestamp:
        i=str(start_timestamp)
        resp = urllib.request.urlopen(server_url + str(i[0:4]) + str(i[5:7]) + str(i[8:10]))
        html = json.loads(resp.read())
        date_flag = html['data']
        print('the date_flag:',date_flag)

        new = pd.DataFrame({"date": str(i[0:4]) + str(i[5:7]) + str(i[8:10]),  "holiday": date_flag}, index=["0"])
        have_holiday =have_holiday.append(new,ignore_index=True)
        have_holiday.to_csv('./data/holiday_china_all.csv')
        start_timestamp = start_timestamp + datetime.timedelta(hours=24)


if __name__=='__main__':
    print('----    首先为字段生成列名    ----')
    #gen_col_name()
    print('----    自动爬取2018年的节假日相关信息（为了防止每次都爬取 爬一次即可）    ----')
    #crawel_holiday()
    print('----    开始构造特征群    ----')
    #make_time_series_fea()


