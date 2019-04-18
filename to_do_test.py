# coding=utf-8
import h5py
from sklearn import preprocessing
from datetime import datetime
import holidays
import urllib.request
import json
import netCDF4 as nc
import pandas as pd
from datetime import datetime
import datetime
import numpy as np
server_url = "http://api.goseek.cn/Tools/holiday?date="
def aaaaa():
    the_test_data = nc.Dataset('./data/ai_challenger_wf2018_testb1_20180829-20181028.nc')
    print('展示测试集其中的variables:',np.asarray(the_test_data['date']))
def bbbb():

    pass
if __name__=='__main__':
    #aaaaa()
    bbbb()