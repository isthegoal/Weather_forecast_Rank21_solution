# coding=utf-8
import pandas as pd
import numpy as np
import datetime
import h5py
from sklearn import preprocessing
import pickle
enc = preprocessing.OneHotEncoder()
'''
进行时序滑动，用于生成一天天用于训练和预测的数据

目标：  每一行形式为....    （前40个小时的历史观测数据，从一个点开始的三个小时的观测数据，从一个点开始的37小时睿图数据，时间点数据，站点标识，要预测的未来的第几个时间  标签   

第一步，对于每个时间点，先划取出这个时间点前40小时   开始的后3小时   把
'''
def  do_slice_big():
    print('4444')
    print('------------------   对时序进行获取   -------------------')
    ruitu_data=pd.read_csv('./data_process/process_data/do_fill_na_ruitu.csv')
    gc_data=pd.read_csv('./data_process/process_data/do_fill_na_gc.csv')

    #对日期转换下，不然认为是string型的
    ruitu_data['time'] = pd.to_datetime(ruitu_data['time'])
    gc_data['time'] = pd.to_datetime(gc_data['time'])

    ruitu_data.pop('Unnamed: 0')
    gc_data.pop('Unnamed: 0')
    air_min_date=np.min(ruitu_data['time'])
    air_max_date=np.max(ruitu_data['time'])
    #分别找到最早能开始的，和最晚截止的（因为最在的前面的时序用不了，最后面的拿来预测也不用能来构建， 所以从中找到合适的时间点）
    print('所有时间点最小的是：',air_min_date)
    print('所有时间点最大的是：',air_max_date)
    air_early_start_time=air_min_date+datetime.timedelta(hours=120)
    air_last_end_point_time=air_max_date-datetime.timedelta(hours=80)  #使用倒数第36小时不合适，因为在训练时候会有目标变量为空的(72小时合适，但是为了做实验，我打算就用10天吧，总27*24那就减小480)
    print('时间点上，最早开始时间为：',air_early_start_time)
    print('时间点上，晚结束时间为：',air_last_end_point_time)
    print('------------------   开始滑动把原始信息，使用stack堆成一行  -------------------')

    guance_before_40_data = []
    guance_hou_37_data = []
    ruitu_hou_37_data = []
    the_big_tog_data = []
    while air_early_start_time <= air_last_end_point_time:
        #对于每个时刻
        for station in [90001, 90002, 90003, 90004,90005,90006,90007,90008,90009,90010]:
            #对于每个站点，分别进行下面的过程
            print('---------   获取这个时间点的前120个小时的观测数据       ----------')
            the_before_start_slide_date=air_early_start_time+datetime.timedelta(hours=-120)
            the_first_range_data=gc_data[(gc_data['the_station']==station)&(gc_data['time']>=the_before_start_slide_date)&(gc_data['time']<air_early_start_time)]
            #print('检查下是否  发生时间顺序错乱',the_first_range_data['utc_time'])
            print(the_first_range_data)
            the_first_range_data.pop('time')
            the_first_range_data.pop('the_station')
            print('需要的第一个列名列表： ',the_first_range_data.columns)
            first_data=list(np.hstack(the_first_range_data.values))
            #print('前144个小时对应的数据维度为：',len(first_data))
            #guance_before_40_data.append(first_data)

            print('---------   获取这个时间点的开始的37个小时睿图的数据   ----------')
            the_last_slide_date=air_early_start_time+datetime.timedelta(hours=36)
            the_secong_range_data=ruitu_data[(ruitu_data['the_station']==station)&(ruitu_data['time']>=air_early_start_time)&(ruitu_data['time']<=the_last_slide_date)]
            #print('检查下是否  发生时间顺序错乱',the_first_range_data['utc_time'])
            the_secong_range_data.pop('time')
            the_secong_range_data.pop('the_station')
            print('需要的第三个列名列表： ',the_secong_range_data.columns)
            second_data=list(np.hstack(the_secong_range_data.values))
            #print('前144个小时对应的数据维度为：',len(first_data))
            #ruitu_hou_37_data.append(second_data)
            print('---------   获取这个时间点的开始的37个小时观测的数据   ----------')
            the_last_slide_date=air_early_start_time+datetime.timedelta(hours=36)
            the_third_range_data=gc_data[(gc_data['the_station']==station)&(gc_data['time']>=air_early_start_time)&(gc_data['time']<=the_last_slide_date)]
            #print('检查下是否  发生时间顺序错乱',the_first_range_data['utc_time'])
            the_third_range_data.pop('time')
            the_third_range_data.pop('the_station')
            print('需要的第二个列名列表： ',the_third_range_data.columns)
            third_data=list(np.hstack(the_third_range_data.values))
            #print('前144个小时对应的数据维度为：',len(first_data))
            #guance_hou_37_data.append(third_data)

            print('---------   进行信息的堆积（时间点、站点、前40(改120)个小时的观测数据、时间点开始的37个小时观测的数据、时间点开始的37个小时睿图的数据）   ----------')
            the_big_together = np.hstack((air_early_start_time,station,first_data, second_data, third_data))
            the_big_tog_data.append(list(the_big_together))
        air_early_start_time = air_early_start_time + datetime.timedelta(hours=1)
    last_big_dataset=pd.DataFrame(the_big_tog_data)
    print('堆积出的大Dataframe:', last_big_dataset)
    last_big_dataset.to_csv('./data/do__slice_data.csv')
def  do_flag_slice():
    '''
    进行标签的切分，主要是最后37个睿图数据进行划分为3个小时是已知的，34个小时是位置的，对于位置的项进行分离成不同时间点标签，三种不同预测变量两种方式。
    对1768个维度进行解析   1+ 1 + 9*120 + 37*29 + 9*37   =1+1+1080+1073+333  =2488（元40小时 1768）                 9*34=306

    10月27号,重新建模：
        现在表示要预测未来33个小时，前面4个小时(3\4\5\6)都是一致的，而后面的33个小时是要预测的。
        现在要构造出的行为:   所属站点（使用直接原型、不做one-hot编码）、前面的40个小时的特征+未来37个小时的瑞图信息+预测点开始的4个小时特征、预测标示、对应小时的真是标签
           1+40*9+37*29+4*9+1+1                33*9=297


    意识到一个问题，我应该在此处构建华东统计特征，而不是生成完之后再构建，真的是，这样的话用生成后的构建统计特征得20个小时，而在这里只需要1个小时不到。
    '''
    print('5555')
    last_big_dataset=pd.read_csv('./data/do__slice_data.csv')
    last_big_dataset.pop('Unnamed: 0')
    #把前面的时间点过滤掉、站点数据ont-hot下
    the_common_data=last_big_dataset.ix[:,2:-297]
    #print(last_big_dataset.head())
    the_date=last_big_dataset.ix[:,0]
    output = open('./feature_data/the_time.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(the_date, output)



    the_station_forma = list(last_big_dataset.ix[:,1])
    #print('经过独热编码后的数据：',the_station_forma)
    #单独获取后面34小时的观测点数据
    the_flag_data=last_big_dataset.ix[:,-297:]

    print('---------   分别分离出t2m_obs  rh2m_obs  w10m_obs  （位于观察序列的第2个、第4个、第5个）----------')
    t2m_obs_clo=[]
    rh2m_obs_clo=[]
    w10m_obs_clo=[]
    number = 2487-8-32*9
    while(number <= 2487):
        t2m_obs_clo.append(str(number+1))
        rh2m_obs_clo.append(str(number + 3))
        w10m_obs_clo.append(str(number + 4))
        number = number + 9

    t2m_all_data = []
    rh2m_all_data = []
    w10m_all_data = []
    print('---------   对于每个变量构造新的行组合（用于构建时间属性的时间点、onehot站点标识[10]、前面的观察特征和3小时观察特征[3*9+9*40]、37小时睿图特征、启动时间特征、实际预测特征）  ----------')
    print('-----  预测t2m_obs模型需要的数据  -----')


    print('--------------     构造特征列名      ---------------')
    the_common_data_col=[]
    ruitu_columns = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M', 'v10m_M', 'SWD_M', 'GLW_M',
                     'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M', 'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M',
                     'wspd975_M', 'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M', 'Q850_M',
                     'Q700_M', 'Q500_M']
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']
    for i in range(0,120):
        for j in guance_columns:
            the_common_data_col.append('guance_before_hour_'+str(40-i)+'_'+j)
    for i in range(0,37):
        for j in ruitu_columns:
            the_common_data_col.append('ruitu_future_hour_'+str(i)+'_'+j)
    for i in range(0,4):
        for j in guance_columns:
            the_common_data_col.append('guance_future_hour_' + str(i) + '_' + j)
    the_common_data=pd.DataFrame(the_common_data.values,columns=the_common_data_col)
    print('总行数：',the_common_data.iloc[:,0].size)
    ii=0
    for index,row in the_common_data.iterrows():


        print('--------------     构造时间特征      ---------------')
        #算了，不想转移时序时间特征了，放在 feture_gen中的make_time_series_fea中了



        '''
          在每个前面都加入统计特征，这样替代后面的那种费时方式。
             构建的统计特征包括：
                 前前面9个指标的前12个小时、前24小时、前36小时的均值，最大值，最小值.三个指标均值的比值
                 共 3*3+3*3+3*3+2+2+2
                 
          我们主要从the_common_data代表的前面360列数据中构造时序，其中
                 
          '''

        print('--------------     构造统计特征      ---------------')
        ii=ii+1
        print('数量：', ii)
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

        t2m_13=mean_t2m_24/mean_t2m_12
        t2m_35=mean_t2m_36/mean_t2m_24
        t2m_57=mean_t2m_72/mean_t2m_36
        t2m_79=mean_t2m_108/mean_t2m_72

        rh2m_13=mean_rh2m_24/mean_rh2m_12
        rh2m_35=mean_rh2m_36/mean_rh2m_24
        rh2m_57=mean_rh2m_72/mean_rh2m_36
        rh2m_79=mean_rh2m_108/mean_rh2m_72

        w10m_13=mean_w10m_24/mean_w10m_12
        w10m_35=mean_w10m_36/mean_w10m_24
        w10m_57=mean_w10m_72/mean_w10m_36
        w10m_79=mean_w10m_108/mean_w10m_72

        list_stat_number = [mean_t2m_12, mean_t2m_24, mean_t2m_36, min_t2m_12, min_t2m_24, min_t2m_36, max_t2m_12,
                            max_t2m_24
            , max_t2m_36, mean_rh2m_12, mean_rh2m_24, mean_rh2m_36, min_rh2m_12, min_rh2m_24, min_rh2m_36, max_rh2m_12,
                            max_rh2m_24
            , max_rh2m_36, mean_w10m_12, mean_w10m_24, mean_w10m_36, min_w10m_12, min_w10m_24, min_w10m_36, max_w10m_12,
                            max_w10m_24
            , max_w10m_36,t2m_13,t2m_35,rh2m_13,rh2m_35,w10m_13,w10m_35,
                            ]

        #错了，少写个rh2m_57，算了不管了，有时间再改过来
        list_stat_number.extend([mean_t2m_72,mean_t2m_108,min_t2m_72,min_t2m_108,max_t2m_72,max_t2m_108,mean_rh2m_72,mean_rh2m_108,min_rh2m_72,min_rh2m_108,max_rh2m_72,max_rh2m_108,mean_w10m_72,mean_w10m_108,min_w10m_72,min_w10m_108,max_w10m_72,max_w10m_108,t2m_57,t2m_79,rh2m_57,rh2m_79,w10m_57,w10m_79])





        print('--------------     构造大行      ---------------')
        for clo_num in range(0, 33):

            #print('长度检验 station:',len(the_station_forma[index][450:]))
            print('长度检验 common:',len(list(the_common_data.loc[index])))
            print('长度检验 common:',len(list_stat_number))
            #进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_t2m=np.hstack((the_station_forma[index],list(the_common_data.loc[index]),list_stat_number,[clo_num+1], list(the_flag_data.loc[index,[t2m_obs_clo[clo_num]]])))
            t2m_all_data.append(every_t2m)

        print('-----  预测rh2m_obs模型需要的数据  -----')

        for clo_num in range(0, 33):
            #进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_rh2m=np.hstack((the_station_forma[index],list(the_common_data.loc[index]),list_stat_number,[clo_num+1], list(the_flag_data.loc[index,[rh2m_obs_clo[clo_num]]])))
            rh2m_all_data.append(every_rh2m)

        print('-----  预测w10m_obs模型需要的数据  -----')

        for clo_num in range(0, 33):
            #进行另一方面的平铺，可以在平铺中加入各种特征，现在数据有 有所
            every_obs=np.hstack((the_station_forma[index],list(the_common_data.loc[index]),list_stat_number,[clo_num+1], list(the_flag_data.loc[index,[w10m_obs_clo[clo_num]]])))
            w10m_all_data.append(every_obs)


    print('---------   对三种数据记性保存   ----------')
    f = h5py.File('./feature_data/t2m_all_data.h5', 'w')
    f['t2m_all_data'] = t2m_all_data
    f = h5py.File('./feature_data/rh2m_all_data.h5', 'w')
    f['rh2m_all_data'] = rh2m_all_data
    f = h5py.File('./feature_data/w10m_all_data.h5', 'w')
    f['w10m_all_data'] = w10m_all_data


    pass
if __name__=='__main__':
    print('---------   大型滑动   ----------')
    #do_slice_big()
    print('---------   小型滑动，产生可以训练的时间点数据   ----------')
    do_flag_slice()
    #得到每行数据，可以进行特征工程了



