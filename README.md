# 竞赛地址
    https://challenger.ai/competition/wf2018
    
# 数据集
     链接：https://pan.baidu.com/s/1dOkjWKtGOf31xDeJipRxeA 
     提取码：flkv 
     
# 代码结构
(1).pipeline.py : 主运行文件。通过该文件调用数据抽取、预处理、时序滑窗、特征构建、训练和预测模块自动处理，使用pipeline的方式可以方便维护方案管理和美化程序结构。

(2)data_process/data_extract.py:数据抽取文件。本赛事直接提供的是nc睿图数据文件，包含了大量的描述和数值信息（可详见data_exper/EDA.ipynb中的数据分析），所以需要进行信息和特征的抽取，在该文件下会分别对睿图数据和观测表数据抽取一定指定时间下数据。

(3)data_process/data_deal.py:这里主要对异常时间进行重新插值的处理

(4)data_process/do_slice.py:构建时序滑窗，针对每个要预测的天气预报时刻，分别构建（前40个小时的历史观测数据、从一个点开始的三个小时的观测数据、从一个点开始的37小时睿图数据、时间点数据、站点标识、要预测的未来的第几个时间、标签)作为一个样本进行时序的搭建。

(5)Feature_engine/feature_gen.py：这里主要构建大量时序特征和统计特征，用来挖掘时序上的趋势关系和特征之间的关联关系，并且根据时序信息构建了时间属性特征，构建的特征可参看py文件中的说明。

(6)model/train_lightGBM.py:主要使用了LightGBM模型，并在进行模型训练前进行了一定的特征降维，不断尝试降低低重要度特征的数量来降低冗余特征所造成的负面影响，具体可详见程序中的说明。

# 思考
（1）这场比赛因为睿图数据较难处理的问题，唬住了很多人，导致最后只有将近60个队伍完成提交，其实只要做好EDA数据分析，对数据和问题进行清晰的认识后，基本的思路和方案都是可以了然而知的。

（2）因为本人只是在决赛前三天参赛的，所以有很多方案并没有很好的尝试，包括没有尝试神经网络的效果，个人认为本赛题提供的数据集是够大的，提供了过去三年的数据进行训练，所以其数据还是能够很好代表本问题的分布情况的，所以如果使用Seq2Seq模型，并进行一定trick的处理后，是能够发挥其优势，拥有较好的效果的。

（3）项目的代码将近3000行，但是写了很多注释，很适合新手学习对时序特征的处理，如果对代码有问题，或者有心一起交流，请加QQ:1091714856

（4）特征方面并没有做够，还是有很多特征没有尝试的，如下为部分特征的构建，程序中都附有大量注释。
        
       #为工作日
       is_workday.append(1)
       is_weekday.append(0)
       is_holiday.append(0)
        #为休息日
        is_workday.append(0)
        is_weekday.append(1)
        is_holiday.append(0)
        #为休息日
        is_workday.append(0)
        is_weekday.append(0)
        is_holiday.append(1)
        #24小时后是否为工作日
        is_workday_after_24.append(1)
        is_weekday_after_24.append(0)
        is_holiday_after_24.append(0)
        #4小时后是否为休息日
        is_workday_after_24.append(0)
        is_weekday_after_24.append(1)
        is_holiday_after_24.append(0)
        #4小时后是否为休息日
        is_workday_after_24.append(0)
        is_weekday_after_24.append(0)
        is_holiday_after_24.append(1)
        #时序特征    分别构建前面9个指标的前5个小时、前12个小时、前24小时、前36小时的均值，最大值，最小值， 比值特征
        mean_t2m_12 = np.mean(t2m_data_fm.loc[index, the_before_12_hour_t2m_obs_list])
        mean_t2m_24 = np.mean(t2m_data_fm.loc[index, the_before_24_hour_t2m_obs_list])
        mean_t2m_36 = np.mean(t2m_data_fm.loc[index, the_before_36_hour_t2m_obs_list])
        # min_t2m_12 = np.min(t2m_data_fm.loc[index, the_before_12_hour_t2m_obs_list])
        # min_t2m_24 = np.min(t2m_data_fm.loc[index, the_before_24_hour_t2m_obs_list])
        # min_t2m_36 = np.min(t2m_data_fm.loc[index, the_before_36_hour_t2m_obs_list])
        # max_t2m_12 = np.max(t2m_data_fm.loc[index, the_before_12_hour_t2m_obs_list])
        # max_t2m_24 = np.max(t2m_data_fm.loc[index, the_before_24_hour_t2m_obs_list])
        # max_t2m_36 = np.max(t2m_data_fm.loc[index, the_before_36_hour_t2m_obs_list])
        the_before_12_hour_rh2m_obs_list=['guance_before_hour_'+str(i)+'_rh2m_obs' for i in range(1,13)]
        the_before_24_hour_rh2m_obs_list=['guance_before_hour_'+str(i)+'_rh2m_obs' for i in range(1,25)]
        the_before_36_hour_rh2m_obs_list=['guance_before_hour_'+str(i)+'_rh2m_obs' for i in range(1,37)]
        mean_rh2m_12 = np.mean(t2m_data_fm.loc[index, the_before_12_hour_rh2m_obs_list])
        mean_rh2m_24 = np.mean(t2m_data_fm.loc[index, the_before_24_hour_rh2m_obs_list])
        mean_rh2m_36 = np.mean(t2m_data_fm.loc[index, the_before_36_hour_rh2m_obs_list])
        min_rh2m_12 = np.min(t2m_data_fm.loc[index, the_before_12_hour_rh2m_obs_list])
        min_rh2m_24 = np.min(t2m_data_fm.loc[index, the_before_24_hour_rh2m_obs_list])
        min_rh2m_36 = np.min(t2m_data_fm.loc[index, the_before_36_hour_rh2m_obs_list])
        max_rh2m_12 = np.max(t2m_data_fm.loc[index, the_before_12_hour_rh2m_obs_list])
        max_rh2m_24 = np.max(t2m_data_fm.loc[index, the_before_24_hour_rh2m_obs_list])
        max_rh2m_36 = np.max(t2m_data_fm.loc[index, the_before_36_hour_rh2m_obs_list])
        the_before_12_hour_w10m_obs_list=['guance_before_hour_'+str(i)+'_w10m_obs' for i in range(1,13)]
        the_before_24_hour_w10m_obs_list=['guance_before_hour_'+str(i)+'_w10m_obs' for i in range(1,25)]
        the_before_36_hour_w10m_obs_list=['guance_before_hour_'+str(i)+'_w10m_obs' for i in range(1,37)]
        mean_w10m_12 = np.mean(t2m_data_fm.loc[index, the_before_12_hour_w10m_obs_list])
        mean_w10m_24 = np.mean(t2m_data_fm.loc[index, the_before_24_hour_w10m_obs_list])
        mean_w10m_36 = np.mean(t2m_data_fm.loc[index, the_before_36_hour_w10m_obs_list])
        min_w10m_12 = np.min(t2m_data_fm.loc[index, the_before_12_hour_w10m_obs_list])
        min_w10m_24 = np.min(t2m_data_fm.loc[index, the_before_24_hour_w10m_obs_list])
        min_w10m_36 = np.min(t2m_data_fm.loc[index, the_before_36_hour_w10m_obs_list])
        max_w10m_12 = np.max(t2m_data_fm.loc[index, the_before_12_hour_w10m_obs_list])
        max_w10m_24 = np.max(t2m_data_fm.loc[index, the_before_24_hour_w10m_obs_list])
        max_w10m_36 = np.max(t2m_data_fm.loc[index, the_before_36_hour_w10m_obs_list])

