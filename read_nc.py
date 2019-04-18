# coding=utf-8
import netCDF4 as nc
import sys
import glob
import csv
'''
既然 nc 可以用来一系列的数组，所以经常被用来存储科学观测数据，最好还是长时间序列的。
试想一下一个科学家每隔一分钟采集一次实验数据并存储了下来，如果不用这种格式存储，时间长了可能就需要创建一系列的 csv 或者 txt 等，
而采用 nc 一个文件就可以搞定，所以会比较方便。
这样就是为了方便花文件处理，才设定出这样的形式进行保存。

'''
def parse_nc():
    the_test_data=nc.Dataset('./data/ai_challenger_wf2018_testa1_20180829-20180924.nc')
    print(the_test_data)
    pass
if __name__=='__main__':
    print('解析nc文件')
    parse_nc()