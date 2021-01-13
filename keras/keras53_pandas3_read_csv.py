import numpy as np
import pandas as pd

# csv를 로드(읽다) .read

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0) 

#  디폴트값 : header = 0 , index_col=None

# header가 없을경우엔     header = None
# index 맨앞이 비어있을떄 index_col=0

print(df)