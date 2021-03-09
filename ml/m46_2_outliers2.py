# outliers1 을 행렬형태로 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],[100,200,3,400,500,600,700,8,900,-1000]])
aaa = aaa.T

def outliers(data_out):
    # data_out = data_out[colum]
    quartile_1, q2, quartile_3 =  np.percentile(data_out, [25,50,75])
    print("1사분위 : ",quartile_1)
    print("q2 : ",q2)
    print('3사분위 : ',quartile_3)
    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

outlier_loc = outliers(aaa)
print('이상치의 위치 : ', outlier_loc)


outlier_loc = outliers(aaa[:,0])
outlier_loc1 = outliers(aaa[:,1])

print('이상치의 위치 : ', outlier_loc)
print('========================================')
print('이상치의 위치 : ', outlier_loc1)

      
