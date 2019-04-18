import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRegressor
from sklearn  import  model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import h5py
import  pickle
import os
from pandas import DataFrame as DF
from sklearn.model_selection import StratifiedKFold

def rmse(y_test, y):
    return np.sqrt(np.mean((y_test - y) ** 2))
def f1(x):
    return np.log(x+1)
def rf1(x):
    return np.exp(x)-1
def get_score(pred, valid_y_exp):
    return np.mean(np.abs(pred - valid_y_exp) / (pred + valid_y_exp) * 2)

def train_lgb_model_t2m(data):
    print('8888')
    '''
    五折交叉检验
    '''
    print('------------  读取数据  ------------')

    #data=pd.read_csv('./feature_data/do_feature_eng_table/t2m_data_tabel.csv',index_col=0)
    y=data.pop('res_value')
    X = data

    print(y)
    print('-----------  开始cv训练  -----------')
    N = 5
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    auc_cv = []

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y[train_index], y[test_index]
        lgb_model = lgb.LGBMRegressor()
        # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
        lgb_model.fit(X_train, y_train)
        pred = lgb_model.predict(X_test)
        tmp_auc = rmse(pred,y_test)
        auc_cv.append(tmp_auc)
        print("valid rmse error:", tmp_auc)

    print('the cv information:')
    print(auc_cv)
    print('cv mean rmse error', np.mean(auc_cv))


    #全量训练并保存模型
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X, y)
    model_file = './model/save_model/lgb_t2m.model'
    with open(model_file, 'wb') as fout:
       pickle.dump(lgb_model, fout)
def train_lgb_model_rh2m(data):
    '''
    五折交叉检验
    '''
    print('------------  读取数据  ------------')
    #data=pd.read_csv('./feature_data/do_feature_eng_table/rh2m_data_tabel.csv',index_col=0)
    y=data.pop('res_value')
    X = data

    print('columns:',X.columns)
    print('columns:',len(X.columns))
    print(y)
    print('-----------  开始cv训练  -----------')
    N = 5
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    auc_cv = []

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y[train_index], y[test_index]
        lgb_model = lgb.LGBMRegressor()
        # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
        lgb_model.fit(X_train, y_train)
        pred = lgb_model.predict(X_test)
        tmp_auc = rmse(pred,y_test)
        auc_cv.append(tmp_auc)
        print("valid rmse error:", tmp_auc)

    print('the cv information:')
    print(auc_cv)
    print('cv mean rmse error', np.mean(auc_cv))

    #全量训练并保存模型
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X, y)
    model_file = './model/save_model/lgb_rh2m.model'
    with open(model_file, 'wb') as fout:
       pickle.dump(lgb_model, fout)

def train_lgb_model_w10m(data):
    '''
    五折交叉检验
    '''
    print('------------  读取数据  ------------')
    #data=pd.read_csv('./feature_data/do_feature_eng_table/w10m_data_tabel.csv',index_col=0)
    y=data.pop('res_value')
    X = data
    print(y)
    print('-----------  开始cv训练  -----------')
    N = 5
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    auc_cv = []

    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X.values[train_index], X.values[test_index], y[train_index], y[test_index]
        lgb_model = lgb.LGBMRegressor()
        # # cv_model = cv(lgb_model, train_data[feature_name], train_label,  cv=10, scoring='f1')
        lgb_model.fit(X_train, y_train)
        pred = lgb_model.predict(X_test)
        tmp_auc = rmse(pred,y_test)
        auc_cv.append(tmp_auc)
        print("valid rmse error:", tmp_auc)

    print('the cv information:')
    print(auc_cv)
    print('cv mean rmse error', np.mean(auc_cv))


    #全量训练并保存模型
    lgb_model = lgb.LGBMRegressor()
    lgb_model.fit(X, y)
    model_file = './model/save_model/lgb_w10m.model'
    with open(model_file, 'wb') as fout:
       pickle.dump(lgb_model, fout)



def train_toge():

    print('-------  使用统一的方式启动训练，这样列名也不用分别放置了。  -------')

    print('----    生成列名    ----')
    all_col_names = list(pd.read_csv('./feature_data/feature_col_name.csv')['0'])
    all_col_names.extend(['is_weekday','is_workday','is_holiday','is_workday_after_24','is_weekday_after_24','is_holiday_after_24','start_hour','the_week',])
    guance_columns = ['psur_obs', 't2m_obs', 'q2m_obs', 'rh2m_obs', 'w10m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs',
                      'RAIN_obs']

    the_columns_name=[]
    for i in range(61,121):
        for j in guance_columns:
            the_columns_name.append(('guance_before_hour_'+str(i)+'_'+j))
    all_col_names=[i for i in all_col_names if i not in the_columns_name]

    print('----    表格构建    ----')
    f = h5py.File('./feature_data/do_feature_eng_table/ver1_t2m_all_data.h5', 'r')
    t2m_data = f['t2m_all_data'].value
    f = h5py.File('./feature_data/do_feature_eng_table/ver1_trh2m_all_data.h5', 'r')
    rh2m_data = f['rh2m_all_data'].value
    f = h5py.File('./feature_data/do_feature_eng_table/ver1_tw10m_all_data.h5', 'r')
    w10m_data = f['w10m_all_data'].value
    t2m_data_fm = pd.DataFrame(t2m_data,columns=all_col_names)
    rh2m_data_fm = pd.DataFrame(rh2m_data,columns=all_col_names)
    w10m_data_fm = pd.DataFrame(w10m_data,columns=all_col_names)


    train_lgb_model_t2m(t2m_data_fm)
    train_lgb_model_rh2m(rh2m_data_fm)
    train_lgb_model_w10m(w10m_data_fm)

if __name__=='__main__':
    train_lgb_model_t2m()
    # train_lgb_model_rh2m()
    # train_lgb_model_w10m()
    # model_lgb_predict(train_air='pm25')


'''


    【1】5个月数据，未调参，5折cv结果为：
      1.9633224218259215
      9.618680703803829
      0.9570037388141832
    【2】尝试参数：
    

'''

