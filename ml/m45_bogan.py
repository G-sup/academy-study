# 결측치 처리법
# 시계열 데이터에서 유리하다. 

from numpy.core.defchararray import index
from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd


datestrs = ['3/1/2021','3/2/2021','3/3/2021','3/4/2021','3/5/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print("=====================================================")

ts = Series([1,np.nan,np.nan,8,10], index =dates)
print(ts)

ts_intp_linear = ts.interpolate() # 판다스에서 제공해주는 보간법 100% 맞는것이 아니기 때문에 골라서 적용해야 한다
print(ts_intp_linear)
