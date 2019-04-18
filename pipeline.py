# coding=utf-8
from data_process.data_extract import nc_to_csv_ruitu,nc_to_csv_gc
from data_process.data_deal import do_data_process
from data_process.do_slice import do_slice_big,do_flag_slice
from Feature_engine.feature_gen import gen_col_name,make_time_series_fea
from model.train_lightGBM import train_lgb_model_t2m,train_lgb_model_rh2m,train_lgb_model_w10m,train_toge


from make_submit.do_new_data_predict_pipeline import do_pipeline

'''
宗旨：
     *少量数据做实验（15天左右）
     *大量跑模型（90天）
     *写pipeline做实际提交
'''
if __name__=='__main__':
    # print('------   首先从nc文件中提取出观察点数据、睿图数据   -------')
    # #历史的1188天的提取注释掉了，因为数据量太大了
    # nc_to_csv_ruitu()
    # nc_to_csv_gc()
    # print('------   数据预处理   -------')
    # do_data_process()
    # # print('------   分离出数据（两次分离，以每个时间点产生形式[站点信息、前40个小时的观测信息和后三个小时的观测信息、后37个小时的睿图信息、预测小时标记、真实变量标签]）   -------')
    # do_slice_big()
    # do_flag_slice()
    # #
    # # print('------   特征工程   -------')
    #gen_col_name()
    #make_time_series_fea()
    # print('------   模型训练   -------')

    train_toge()


    print('------   预测和构造提交文件形式   -------')
    #do_pipeline()


