from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[1800,1,2,3,4,6,7,8,90,100,2000],[100,200,3,400,500,600,700,8,900,-1000,-100]])

#  2차원이상일떄 기준은 열이다

aaa = aaa.T

print(aaa.shape)


outlier = EllipticEnvelope(contamination=0.1) # contamination 오염된 (지정한 임의의 아웃라이어의 퍼센테이지) 통상 0.1정도(10%)로 잡는다

# 가우시안 분포 처리 

outlier.fit(aaa)

print(outlier.predict(aaa))