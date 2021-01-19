import pandas as pd

df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]], columns=list('abcd'),index=('가','나','다'))
print(df)

df2 = df        # 이거는 메모리를 공유 (원본보존x)

df2 ['a'] = 100

print(df2)

print(df)

print(id(df), id(df2)) #판다스는 아이디를 공유


df3 = df.copy() # 복사 하려면 copy 원본을 안건드릴려면

df2['b'] = 333

print('=====================')
print(df)
print(df2)
print(df3)

df = df + 99    # 이거는 메모리를 공유하지 않는다

print('=====================')
print(df)
print(df2)