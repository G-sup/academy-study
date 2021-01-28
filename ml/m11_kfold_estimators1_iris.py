from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
# from sklearn.utils import all_estimators

import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=104)

KFold = KFold(n_splits=5,shuffle=True) # (n_splits=5 : 몇개로 나눌것인가 ,shuffle=False : 순차적)


allAlgorithms = all_estimators(type_filter='classifier')

for (name,algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=KFold) # kfold 대신에 숫자를 넣어도 된다 단 셔플X


        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name,'의 정답률 : \n', scores)
    except:
        # continue
        print(name,'없는 모델')

import sklearn
print(sklearn.__version__) # 0.23.2
